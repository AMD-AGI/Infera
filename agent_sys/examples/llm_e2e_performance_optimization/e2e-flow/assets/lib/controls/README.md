# Control overlays — NONE OF THESE IS AN OPTIMISATION

Three replacements for `srt/layers/sampler.py`, kept so that anyone who doubts
`check_no_regression` can **run** the controls rather than rebuild them. They
exist to test the instrument, not the engine.

**No kernel was installed by any of them. M5.1.1 is untouched. A green run over
these is "the plumbing works and the gate does not hallucinate a difference",
never "module 5 works".**

| | what it is | measured on 047, 2026-09-04 | verdict |
|---|---|---|---|
| `null/` | stock + a marker constant. Semantically identical. | ITL 10.02 ms vs stock 10.14, tps 448.63 vs 448.82 | `same` |
| `degraded/` | stock + a 2 ms sleep in `Sampler.forward` | ITL **10.05 ms** — no change | **a control that failed to be a control** |
| `marked20/` | stock + a 20 ms sleep **and** runtime markers | ITL **33.78 ms** (+233%), tps 421.00 (−6.2%) | `REGRESSED` |

Predictions for all three were committed in `RUN-PLAN.md` (`d71d765`) **before**
the run, with the timeline that makes that auditable.

## Why `degraded/` is kept even though it does not work

**It is the most useful of the three.** It was predicted to regress by +20% and
did not move, and the convenient reading — "the mount failed" — was wrong:

```
in-container hash   3c729c05ec1a  == the overlay's own sha256_patched
the running file    contained the sleep
its .pyc            compiled that minute
```

So a third overlay, `marked20/`, added runtime markers to answer the question
the static evidence could not. **Both fired — import 18 hits, `first_call` 8** —
so `Sampler.forward` *does* execute and a 2 ms CPU sleep in it changes nothing.
The explanation consistent with every measurement is **overlap scheduling**: the
CPU-side sampling for one decode step runs under the GPU compute of the next, so
2 ms fits inside a ~10 ms step and disappears. At 20 ms the CPU becomes the
bottleneck.

**That is why `check_patch_live` now requires a runtime marker by default.**
Perfect static evidence could not distinguish *mounted and never executed* from
*executed and had no effect*, which is the one distinction it exists to draw.

## Why a null control alone would not have been enough

*Known-no-effect in, no effect out* is a **negative control**. A gate validated
only against a null sample has never been shown to **detect** anything: a
null-only experiment prints "no difference" and licenses nothing. It took a
deliberate regression that the gate *failed to see* to discover that the 2 ms
was never a regression at all.

## Using one

Each directory is a `kernel_optimization`'s payload: `sampler.py` is the
replacement and `manifest.json` is its `apply/manifest.json`. `base_sha256` is
pinned to `inferaimage/infera:sglang-local`'s stock file — **re-cut it against
whatever image you are serving**, or `apply_patch` will refuse with a hash
mismatch, which is that gate working.

`expect.source` is `null_overlay` or `degraded_control`, so
`kernel_reconciliation` prints what each is without anyone adding a mechanism.

All three keep every public definition of the file they replace, so `apply_patch`'s
symbol-set check passes them honestly rather than by exemption.

## The negative-test battery beside these — `artefact_neg.py`

`ARMS=<dir> python3 artefact_neg.py` breaks a real artefact six ways and checks
the validator refuses. It is here for the same reason the overlays are: **a pass
count cannot tell a well-built check from one grading `nonempty`**, and the next
person to doubt one of these should be able to run the case rather than
reconstruct it.

**What six passing probes license, and what they do not.** They show these
validators detect **the specific breakage injected here**. They do **not** show
they detect the breakage the world produces — m4's standing caveat about their
own re-measurement, in different clothes. A green run is evidence about six
cases, not about the class.

**Three of the probes were wrong first, and they are kept in the file.** Each
returned a confident PASS while changing nothing — the packup has two
`README.md` and `rglob` gave the wrong one; `smoke.json`'s `checks` is a list and
a `dict` type-guard skipped the tamper entirely; needle depths carry `ok` while
the probe set `retrieved`. **A PASS from a probe you have just written is not
evidence**; check the field you edited is the field the validator reads.

`zone_harness.py` is committed alongside because the battery cannot run without
it, and it builds the zone the way `validator/phase.py` does — `env -i` with the
GLOBAL row only, because a run that inherits your shell is a different
experiment and will tell you the subject is fine.
