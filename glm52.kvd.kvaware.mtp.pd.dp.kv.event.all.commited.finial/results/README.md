# Results — which file is which gate

All from the **built image** run (`infera/engine-sglang:merged`, no in-container
patching). Regenerate any of these by following `../REPRODUCE.md`.

| file | gate | result |
|---|---|---|
| `raw/g2fix_needle.builtimage.json` | **G2**, MTP on, official sampling | **5/5**, all `finish=stop`, 73–259 ctok |
| `raw/stress_c16.builtimage.json` | **conc=16** | **64/64 CLEAN**, 0 BAD |
| `raw/stress_c128_osl2048.builtimage.json` | **conc=128** | 247 CLEAN / 7 TAIL_REPEAT / **1 BAD** / 1 WRONG |
| `raw/g2ctl_needle.builtimage.json` | *control*, MTP **off** | 5/5 — see below |
| `raw/stress_c128.builtimage.json` | *superseded*, conc=128 at OSL **1024** | 5 BAD — see below |
| `raw/replay_c1.builtimage.json` | *diagnostic*, the 6 conc=128 failures replayed at conc=1 | **12/12 CLEAN** |

G0 and G1 produce counters rather than JSON; their numbers are in `../README.md`
and reproduced by the commands in `../REPRODUCE.md` §5–6.

## The three non-gate files, and why they are here

They are the evidence for a diagnosis, not results. Without them the reasoning in
`../notes.md` §1 is an assertion.

**`g2ctl_needle` — the control that exonerated the early-send fix.** G2 first read
3/5 with MTP on, matching the mooncake corruption signature. This is the same
prompt, same image, same prefill leg, with MTP off on the decode leg: **5/5**. One
variable, so the prefill path is correct and the failure lives in decode-side
sampling.

**`stress_c128` — the superseded OSL-1024 run.** 5 BAD, and every one of them at
the 1024 cap. Raising OSL to 2048 alone gives `stress_c128_osl2048` with 1 BAD.
Kept because "5 BAD" appears in `notes.md` §1 and the reader should be able to
check it.

**`replay_c1` — the load-dependence test.** The six prompts that failed at
conc=128, replayed twice each at conc=1. Prompt content is a pure function of
`idx+salt`, so these are byte-identical to what failed. **12/12 CLEAN**, which is
what established the failures as cap-related rather than KV-related.

## Reading the verdicts

- `CLEAN` — coherent, correct answer, self-terminated.
- `TAIL_REPEAT` — needle retrieved and `finish=stop`; only the post-`</think>`
  tail loops. **Not a failure.**
- `WRONG` — coherent and self-terminated but the expected number is absent. In
  `stress_c128_osl2048` the single case reads `33663` for `33643`: a model
  misread, not corruption.
- `DIGIT_LOOP` / `CORRUPT_REASONING` — the BAD tally, the criterion for this gate.

The classifier was corrected in the predecessor run after two defects that between
them hid the very failure mode the gate exists to catch; see that kit's
`notes.md` §6.
