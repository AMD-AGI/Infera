# seed_table — mock the upstream profiling handoff

Republish a Magpie gap-analysis CSV as a `kernel_table` handoff, so this package
runs standalone before `profiling-demo` exists.

## What it does

1. Read the CSV named by `$AD_SEED_CSV`.
2. Accept either header shape — six columns from a plain Magpie run, or nineteen
   when Magpie ran with `--find-kernel-sources` — and record which one it got.
3. Copy the CSV verbatim into `items/gap_analysis/`.
4. Write `items/text.json`: one record per kernel, numbers parsed.

## Why it exists

agent_sys has no route for injecting a handoff from outside the graph. A handoff
is produced by a task or it does not exist, so a package that wants to be
runnable on its own has to produce its own input.

## How to remove it

When `profiling-demo` lands, its `kernel_scan` produces the same
`kernel_table` kind. Then:

1. Delete `steps/seed_table.yaml` and `assets/seed_table.task/`.
2. Delete the `{closure: seed_table, froms: []}` entry from `main.yaml`.
3. Change `rank`'s subgraph entry to `froms: [kernel_scan]`.

Nothing else moves. The kind lives in `shared.yaml` precisely so that it
survives this deletion.

## Watch out

The default CSV is a **GLM-5.2 1P1D decode profile**, not GLM-5.3-Flash. Its
shapes are real and exercise the pipeline correctly, but its operator mix is not
the target model's — GLM-5.3-Flash serves DSA through TileLang, KDA through
Triton, and MoE through the Triton runner. The README this task writes states
that provenance, so a downstream reader is not left to infer it from a path.
