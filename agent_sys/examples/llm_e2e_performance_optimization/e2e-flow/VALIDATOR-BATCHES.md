# Validator batches — which ones are on, why, and what is still off

**Written 2026-09-05 by m5, because the batch membership existed nowhere except
in two messages and one `/tmp` file.** m2 needed it, could not find it, and
reasonably worked around it by enabling all 22 and reasoning that the two
known-refusal validators are unreachable before `apply_patch`. That reasoning
was sound and the next person will not repeat it — they will guess.

This is the *二分打开* list: the user's instruction is to get a chain walking with
validation off, then re-enable in batches, easiest first, and file the hard ones
rather than fight them.

## How to use it

```sh
python3 assets/lib/make_debug_package.py --out <repo>/e2e-flow-keep --keep "$(<the list below>)"
```

`--keep` is **per validator name, not per (kind, validator)**. That is the
property that decides batch membership: a validator that passes on fourteen
kinds and refuses on one costs you the whole run at the kind that refuses. So a
validator only joins a batch when it passes **everywhere**, not on balance.

**Re-verify `check_deploy_serves` is unwired after regeneration** rather than
trusting `--keep` to have excluded it — m1 caught that one.

## The batches, as run

### keep11 — first batch, 2026-09-05

```
check_acceptance,check_bench_report,check_bench_result,check_command_parses,
check_deploy_kit,check_identity_resolved,check_kernel_table,check_overlay_applies,
check_packup_shape,check_trace_coverage,check_worklist_shape
```

Result: chain completed, **17/17 tasks, 20 real verdicts, 0 non-PASS**.

### keep12 — adds `check_environment`

`keep11` + `check_environment`.

Held out of `keep11` deliberately: at that point it refused on `e2e_packup` and
`kernel_optimization`, and per the rule above that would have stopped the chain
at **stage 4, before `apply_patch`**. Added once both artefacts carried a record.

Result: chain completed, **17/17, 35 real verdicts, 0 non-PASS**,
`check_environment` **15/15 kinds**.

**That run used a patched test root** (`/home/yihou/cheat_for_mock.m5fix.*`),
not the corpus everyone else uses. Do not quote its 35 against a run on the
shared corpus without saying so.

### keep13 — adds `check_measurement_order`

`keep12` + `check_measurement_order`. **Result unknown** — the run was killed at
`integrate_and_verify` before it graded anything. The question it exists to
answer is still open; see below.

## What is still off, and why

| validator | why | owner |
|---|---|---|
| `check_no_regression` | refuses correctly — provenance gap between a real stock arm and corpus m2 numbers | nobody; it is right |
| `check_patch_live` | refuses correctly — no `runtime_marker`; M5.1.1 | with the user |
| `check_optimization_shape` | `packup_noop_<date>` breaks the required `packup_<YYYYMMDD>` | leader's builder |
| `check_workset_shape` | sealed workset has no evidence block | m3 |
| `check_profiling_evidence` | ranking and trace are different captures (419218 vs 826040 events) | leader's grafted corpus |
| `check_measurement_order` | **unknown, not failing** — see below | m5 |
| `check_deploy_serves` | real bring-up + 180 s load; hung a run 33 min | not batchable |
| `check_speedup_substantiated` | re-measures against a live container | `cost: gpu_hours` |
| `check_workset_runs` | re-measures against a live container | `cost: gpu_hours` |

**`check_measurement_order` is the cheapest remaining win and is not a
refusal.** The offline probe stages **one kind per validator**, and this one
needs **both arms** — *"the two kinds are produced by ONE task (M5.2), so they
arrive together"*. So the probe cannot grade it and never could. It may already
pass in situ; nobody has looked. Pre-register it as **expected PASS** before the
run that answers it, so a refusal is a finding rather than an argument.

## Grading a batch without a node

```sh
python3 assets/lib/probe_validators.py --run <a completed run> \
    --var expect_ranks=<the run's value> --var adhoc_cases=<the run's value> \
    --var bench_rounds=<the run's value> --jobs 6
```

**Pass the run's own `--var`s.** The probe substitutes from `MOCK_VARS`, which is
tuned to the 09-02 corpus (`expect_ranks=2`, `adhoc_cases=0`). Graded against a
run that used other values it invents refusals that belong to nothing — it
produced three that way before `--var` existed (`91b173b`). A checker that
disagrees with the launch line reports the difference as the artefact's defect.

**And check the version.** `v0` being empty is the norm for any handoff whose
task retried; grade the populated version, not the first one. Two conclusions
were drawn from empty `v0` directories in one hour before that was noticed.
