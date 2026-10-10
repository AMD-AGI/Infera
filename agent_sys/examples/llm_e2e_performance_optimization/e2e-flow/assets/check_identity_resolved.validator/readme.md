# `check_identity_resolved` — trustworthiness, strong

Every candidate carries a resolution level, and an unresolved one says why rather
than guessing.

## Read `min_resolve_ratio: 0.0` first

It is deliberate and it is not a disabled check.

The evidence available to `identify` is a device symbol. For a Tensile GEMM
dispatched from C++ there is no Python frame to find, and a generated symbol
frequently appears nowhere in the checkout — the sealed artefact's own case is
`mfma_moe1_silu_mul_afp4_wfp4_bf16_g1u1`, whose generator never emits `_g1u1` at
all. A floor above zero would fail a correct analysis on exactly the kernels the
next stage excludes anyway, and the pressure it creates is to write a confident
`source_file_path` instead of an honest `resolution_hint`. That is the opposite
of what this validator is for.

**What is graded is that uncertainty is modelled, not that it is absent.**

## What it checks

1. `items/text.json` validates against
   `assets/schemas/operator_identity.schema.json`, which carries the modelling
   rules: an operator with no `source_file_path` must have a `resolution_hint`
   or an `excluded_reason`; `agent_recovered` must have a hint; `symbol_search`
   must record the probe.
2. `items/schema` is that file byte for byte (CONTRACT.md §3.4).
3. Every non-empty `image_repo_path` is a key of `container_root_placeholders`.
   An undeclared placeholder expands to nothing and the consumer reads the wrong
   tree. **Empty is permitted** — it means no owner rule matched, which happens
   for real and is a state to model rather than forbid.
4. `kernel_id` and `logical_operator` are each unique. The second matters
   because `logical_operator` becomes a directory name in the workset, so two of
   them collide silently.
5. `summary.resolved` and `summary.resolve_ratio` are **recomputed** and must
   agree with `operators`. An operator counts as resolved only when a file was
   found by a method that finds files — a `source_file_path` filled in under
   `agent_recovered` is a guess wearing a fact's clothes.
6. The recomputed ratio clears `min_resolve_ratio`.

Rule 5 recomputes for the reason `check_workset_runs` recomputes a weighted mean
and `check_no_regression` recomputes a verdict: a stored figure that does not
follow from the list can only come from the record having been edited after it
was produced.

## What it reports and never grades

The per-operator resolution level and, for each unresolved one, its hint. This is
the number a reader wants and the number that must not become a target.
