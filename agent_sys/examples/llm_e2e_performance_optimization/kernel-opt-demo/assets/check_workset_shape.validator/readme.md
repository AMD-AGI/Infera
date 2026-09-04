# check_workset_shape

**Claim:** the workset carries every file an optimizer needs, its driver
implements the measurement contract, and it declares at least three correctness
cases.

`completeness` / `strong` / `cost: seconds` / `logic_source: external_static`.

## Why `strong` without qualification

Every rule is decided by looking at a file that is there or is not, or a string
that occurs or does not. Nothing is judged and nothing is sampled. This body
cannot be *approximately* right about whether `driver.py` exists.

## The rules, in full

1. Exactly **one** directory under `items/codes/`. Zero means the producer wrote
   it somewhere the `code` content type does not put it; two means a consumer
   has to guess which one to optimize.
2. Every path in `args.required_files` exists **and is non-empty**.
3. `kernel/driver.py` contains each of `args.required_driver_tokens` —
   `case_ms:`, `case_snr:`, `SNR:`, `--bench-mode`.
4. `driver.py`'s `_CASES` tuple declares at least `args.min_correctness_cases`
   shapes.
5. `baseline_measurement.md` has ≥5 content lines and contains at least one
   digit.
6. None of `README.md`, `program.md`, `integration.md` carries `TODO`, `TBD`,
   `FIXME`, `XXX` or `to be filled in`.

"Content line" means: non-blank, not a heading, not a fence marker.

## Why rule 3 is worth its own check

The entire optimization loop reads correctness and timing off `driver.py`'s
**stdout**. A workset whose driver does not print `case_ms:` cannot be
benchmarked at all — and the failure does not appear at load time. It appears
hours later, as a campaign that ran to completion and measured nothing. Four
substring checks buy that back for nothing.

## Where it runs

**Both phases**, and that is the point. It runs in `publish_workset`'s output
phase and again in `optimize_kernel`'s input phase, over the same kind. A
workset that lost a file between publication and consumption is a real failure
and the second run is what would catch it. It is also why `entry.sh` writes both
environment fallbacks: the input phase gets the GLOBAL row and never
`AGENT_SYS_TASK_PACKAGE`.

## What it cannot catch

Stated because a validator that overclaims is worse than none.

- **It does not run the driver.** A driver that prints the right tokens and
  measures the wrong thing passes here.
- **It does not verify the baseline numbers.** It checks that a cross-check
  against a profile figure is present and claimed, not that the claim is true.
- **It does not check the kernel is correct or even that it imports.** Nothing
  here touches a GPU.

Establishing that the driver measures the traced kernel is `optimize_kernel`'s
first job, and this body is not a substitute for it. Rule 5 exists to make the
*absence* of that evidence visible, not to confirm its presence.
