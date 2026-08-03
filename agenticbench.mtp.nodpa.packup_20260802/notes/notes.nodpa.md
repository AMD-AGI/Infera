# noDPA — what went wrong, and what each cost

Five things went wrong. Three were caught before a measured window; one cost a
full 33-minute rerun; one is a permanent artifact loss. Each is stated as
**what / why / how / context**.

The expensive one is §3 — a flag whose log message says "adjusted to 8192" when
the code says `// dp_size`. It is also the one that produced the most useful
by-product, so it is worth reading even if nothing else here is.

---

## 1. The deployment was gone before the run started

**What.** lat1's two jobs (24300 / 24301) had hit their 24 h walltime and were
released at **08:36 UTC**, ~3 h before this session started. `squeue -u $USER`
returned nothing.

**Why it matters.** lat1's environment.md says "same live deployment as Case A" —
that property does not transfer here. This run is a **cold rebuild on two new
nodes**, so the comparison rests on same-branch/same-commit/same-flags rather
than same-process.

**How.** Held two new nodes (28490 / `crsuse2-m2m-231`, 28485 / `crsuse2-m2m-276`),
rebuilt the image on both from `src.tar`, re-verified everything from scratch.

**Context.** Cost ~1 h. The upside is genuine: a cold radix tree and a cold kvd
make seed contamination structurally much less likely, and the bytecode gate +
`server_args=` checks re-proved the whole stack rather than assuming it.

**Corollary for anyone reproducing:** four nodes bounced with
`JobHoldMaxRequeue` before two held. Catch the node each held job landed on,
`--exclude` it, retry — and confirm the hold with `spur exec <job> true`, because
a job can read `RUNNING` for one poll and still requeue.

---

## 2. `stage_source.sh` swept the packup directories into the build context

**What.** `src.tar` went from 4.6 MB (the 2026-08-01 build) to **18 MB**.

**Why.** Its exclude regex listed `glm52.`, `kvaware_kvd_pr.`, `liying_rest_pr56.`,
`merge_kvaware_mtp_pd.`, `work.` — but the three `agenticbench.mtp.*` packup dirs
were created *after* that regex was written, so they matched nothing and were
tarred into the docker build context.

**How.** Added `agenticbench\.` and `partial\.debug` to the exclude. Back to 12 MB.

**Context.** Harmless to correctness — docker would have ignored the files — but
it inflates every build and would grow without bound as more packups land. The
fixed script is in `scripts/`.

---

## 3. `chunked-prefill-size`: misread the mechanism, and it cost a 33-minute rerun

**What.** The measured window was run twice. The first run used a global per-step
budget of 8,192 — **⅛** of what the lat1 arm it is compared against was getting.

**Why.** The stock leg script's `else` branch hardcoded `CHUNK=8192` for `DPA=0`,
against the `DPA=1` default of `8192 × TP = 65536`. That much was spotted at design
time. What was then got *wrong* is what the engine does with the value.

Both legs log:

    WARNING: DP attention is enabled. The chunked prefill size is adjusted to
    8192 to avoid MoE kernel issues.

I read "adjusted to 8192" as a **clamp** — i.e. that the engine ignores the passed
value whenever DPA is on, so lat1's *effective* chunk was 8,192 and matching it
meant passing 8,192. On that reading the first prefill boot at 65,536 looked like
the error, and it was "corrected" to 8,192 before the measured window.

**It is not a clamp.** `server_args.py:4902`:

```python
if self._resolved().enable_dp_attention:
    self.chunked_prefill_size = self.chunked_prefill_size // self.dp_size
```

A **division by `dp_size`**. The 8,192 is `65536 // 8` and it is **per rank**; the
global budget is still 65,536. With DPA off there is no division, so passing 8,192
gives 8,192 for the whole machine.

**How.** Read the source instead of the log text, rebooted prefill at
`CHUNK=65536`, reran probe + the full 30-minute window. `server_args=` then reads
`chunked_prefill_size=65536` with `enable_dp_attention=False`
(`../env/chunk65536_prefill_server_args.txt`).

Independent cross-check that does not depend on reading flag semantics at all —
`#new-token`, the tokens actually processed per prefill step:

| arm | modal `#new-token` | meaning |
|---|---:|---|
| lat1 (dp8) | 8,192 | per rank, **× 8 ranks concurrently** |
| noDPA-8K | 8,192 | the entire machine |
| noDPA-65K | **max 65,536** | impossible under an 8,192 budget |

**Context.** Cost ~50 min of cluster time. The 8,192 run is **kept**, as
`../results/chunk8192_ARM/`: the two noDPA arms differ only in chunk, so together
they measure the chunk effect in isolation — **nil, 0.98–1.06× across seven bins**.
That null result is what licenses attributing the remaining 1.65–1.93× to DPA, and
it could not have been assumed in advance. The mistake produced the control.

**Generalisation worth carrying: read the source for a flag whose log message is
ambiguous.** "adjusted to N" does not say whether N is per-rank or global, and the
whole experiment turned on which one it was.

## 4. `random_seed` exceeded numpy's range — again

**What.** The first probe died at startup:

    ValueError: Seed must be between 0 and 2**32 - 1

**Why.** I picked `20260802001` / `20260802002` by appending digits to lat1's
`20260802`, which pushes them to ~2.03e10 against a bound of 4.29e9. lat1's own
kit documents this exact defect (its `20260802999` failed the same way); I
reproduced it while trying to avoid its *other* defect.

**How.** Changed to `2026080201` / `2026080202` (~2.03e9), and added an assertion
to the workflow rather than trusting eyeballs (the MAIN arm later re-seeded again
to `2026080211` / `2026080212` for the rerun, since the 8,192 arm had by then
warmed the tree with the first pair):

```python
assert int(seed) <= 2**32 - 1
```

**Context.** Cost ~4 min because the probe caught it. In the measured window it
would have cost 33 min. This is the second time this defect has appeared in two
runs — worth a lint rule on the YAML, not just a note.

---

## 5. The GMU 0.80 OOM crash log was destroyed by the reboot

**What.** With `GMU=0.80` (lat1's value) the noDPA prefill leg crashed:

    rocdevice.cpp:3582 ... HSA_STATUS_ERROR_OUT_OF_RESOURCES ...
    Available Free mem : 254 MB
    Fatal Python error: Aborted

Diagnosed live from the surrounding log lines — `token usage: 0.04` (KV pool
empty, so **not** KV exhaustion), `#new-seq: 1`, `#running-req: 0` (one request,
so not batching) — therefore **activation** memory.

**Why the artifact is gone.** `glm52_leg_spur_mtp.sh` redirects with `>`, which
truncates. Rebooting at `GMU=0.70` into the same `nodpa_prefill.log` overwrote the
crash. Confirmed: the current log has `grep -c HSA_STATUS_ERROR_OUT_OF_RESOURCES`
= 0 and only `mem_fraction_static=0.7`.

**How (not) fixed.** Not recoverable. The finding is recorded in
`analysis/nodpa_vs_lat1.md` and `notes/nodpa_design.md` with the verbatim error
and the diagnostic reasoning, and `environment.md` lists the missing artifact as
an explicit gap. **Anyone re-running reproduces it by booting this arm at 0.80.**

**Context.** The finding itself is a real result — *DP-attention off costs
activation headroom* — and it is the reason this arm carries two flag differences
instead of one. The evidence for it being immaterial to the TTFT result
(`token usage: 0.04` on both arms) **is** preserved, in this run's shipped logs.

**Fix for next time:** append (`>>`) rather than truncate, or timestamp the log
filename per boot. A crash log that a retry destroys is the one you most want.

---

## Two false alarms, recorded so they are not re-investigated

### `grep -ci retract` on the decode log reads 1829

It matches the routine per-line field `#retracted-req: 0`. Real retractions are
**zero**:

```bash
strings nodpa_decode.log | grep -c '#retracted-req: [1-9]'   # -> 0
```

### `gate.sh` reported `dp_size=8` on the prefill leg

`gate.sh` takes the log tag as `$1` and defaults to `g1`. Run bare, it read a
previous run's `g1_prefill.log` and reported that run's DPA state — which is
exactly the value this experiment was checking was *absent*. Run it as
`gate.sh nodpa`. Cost 2 min and one moment of alarm.

---

## Needle: 2/5 then 5/5, and why that is not KV corruption

First pass scored 2/5 with the three failures all `finish=length` — the model
generating repetition to the 2048-token cap (`6"6"6"6"…`) rather than wrong
digits. Retest scored **5/5**, every depth returning its exact 7-digit value.

The discriminator is that **the failing depths moved** (5 %, 50 %, 95 % failed then
passed) and every depth answered correctly at least once. KV corruption does not
relocate between runs. The first pass also ran cold (`cached=None`, `5952`,
`29952`…) while the retest was warm (`cached=120000` throughout).

Same signature Case A documented (3/5 then 4/5, failing depths swapped) and
diagnosed as sampling variance at the model's own `temperature 1.0 / top_p 0.95`.
Per the project's standing instruction, this was retested once and dropped.

**Do not "fix" this by setting `temperature: 0`** — greedy decoding on a long
prompt sends GLM-5.2 into repetition and EAGLE amplifies it, producing something
indistinguishable from KV corruption.
