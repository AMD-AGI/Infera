# check_workset_shape — completeness, strong

Every operator directory carries the files KernelForge reads, and the two
exported formats agree with each other.

## Shape, not quality

Whether a driver actually runs, and whether the numbers it prints are right, is
`check_workset_runs`' question. No static check answers it.

The split is deliberate and follows `validator` spec §5.3, which orders a phase
cheap first: this validator is `external_static` / `seconds`, the other is
`external_dynamic` / `gpu_hours`. A workset missing a file fails here, before a
GPU is booked for it.

## What it checks

1. The `reproducible` items — `result`, `env`, `script`, `code` — are present.
2. Every operator directory carries every file in `required_files`, non-empty.
3. `invocation_spec.json` loads, declares `schema_version: 2`, and its `status`
   agrees with `missing_fields`: `partial` needs a non-empty list, `complete`
   needs an empty one.
4. `forge_task.yaml` loads and carries `task_id`, `gpu_target`, `shapes.primary`
   and `targets.snr_db`.
5. The two agree on case count and primary case.
6. `tests/cases.json` has at least `min_cases` entries, each carrying a `CASE_ID`.
7. `scripts/forge_driver.py` parses as Python and mentions `--bench-mode`,
   `--profile-run` and `SNR`, all three of which forge-loop's preflight requires.
8. No file names an absolute path outside the seal's allow-list.

## Why rule 8 exists when the seal does the same thing

The seal reports the first offending path and refuses the delivery, which
surfaces as `output was never delivered` after the work is done. Checking it
here names the file and the line while the workset is still in hand.
