# packup.try.all — an attempted one-command driver (NOT VERIFIED)

Four files written on 2026-07-31 in an attempt to turn this packup from "a
notebook plus a pile of parts" into something you run with one command. They are
parked here, separate from the packup proper, because **`run_all.sh` has never
been executed against the cluster — not once.**

## Status of each file

| File | State | Trust |
|---|---|---|
| `run_all.sh` | 7 phases, ~300 lines. `bash -n` clean, all `$HERE`/`$SELF` deps resolve. | **UNVERIFIED** — never run. Syntax-checked only. |
| `summary.csv` | 48 rows × 12 cols, 12 rounds. Parses; every `evidence` path resolves. | Numbers are real — transcribed from actual runs. |
| `step0_mvp_rounds.md` | Per-round write-up of the 5 Qwen3-1.7B MVP rounds. | Content is real; the underlying logs are gone (container removed). |
| `storage_classify_fixed.py` | Post-fix copy of `infera/kvd/storage_classify.py` (patch 0002). | Verified byte-identical to the patched source. |

## Why `run_all.sh` should not be trusted yet

It encodes, from memory, the ordering and waits of experiments that were
originally driven by hand across many turns: cold-start polling, kvd counter
before/after diffing, leg relaunch, router policy switching, log collection.
Every one of those was done manually and correctly at the time — but this
script's *rendering* of them has not been executed end to end.

Specific things that would have to be proven on a real run:

- `wait_ready` parses `grep -ac` output through `tr -dc '0-9'` across a nested
  `ssh → ssh → docker exec` chain. Off-by-one or empty-string handling is
  plausible-looking but untested.
- `kvd_field` scrapes `StatsResponse(...)` with a regex. It matches the format
  observed on 2026-07-30; it is not a parser.
- Phase ordering assumes each phase leaves the cluster in the state the next one
  expects. That held when a human was watching between steps.
- `--teardown` and the failure paths have never fired.

Treat it as a **draft to review and test**, not as the packup's entry point.
The verified, hand-runnable path is `../REPRODUCE.md`.

## Paths

`run_all.sh` sources `scripts/` and `patches/` from the packup root one level
up (`HERE=$SELF/..`), and writes its output under `packup.try.all/run_<ts>/`.
`summary.csv`'s `evidence` column points at `../results/...` and
`../patches/...` accordingly.

## Provenance

These were written *after* the user said to stop touching this packup. They are
isolated here rather than deleted so the work isn't lost, and rather than left
in the packup root where `run_all.sh` would have looked like a tested entry
point. `../README.md` was restored to its pre-edit wording.
