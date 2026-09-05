# `check_worklist_shape` — completeness, strong

The worklist validates against its schema, its buckets are accounted for, and
every exclusion carries a reason.

Program, not AI. Every rule is decided by a schema that validates or does not, or
by arithmetic over a document that adds up or does not. Nothing is judged.

## What it checks

1. `items/text.json` validates against `assets/schemas/kernel_worklist.schema.json`.
   Every problem is reported at once — one at a time turns a five-field mistake
   into five runs.
2. `items/schema` is that file **byte for byte** (CONTRACT.md §3.4). A carried
   schema that may drift makes the artefact self-describing and describing
   something other than what graded it.
3. The bucket shares sum into `[80.0, 100.5]`, and the bucket counts sum to the
   number of rows. Below the floor the ranking saw a fraction of the work; above
   the ceiling something is counted twice across ranks.
4. Each bucket's declared count equals what `kernels` actually holds for it.
5. `kernel_id` is unique, the selected rows rank `1..N` with no gap or repeat,
   and no more than `top_n` are selected.
6. `summary` agrees with `kernels`.
7. `items/worklist.csv` agrees with `items/text.json` on every row's
   `selected` and `bucket`.

## Why 3 and 7 are here rather than in the schema

A JSON Schema can bound one number and cannot sum a list, and it cannot compare
two documents. Those are the two failures that actually happen: a `buckets` block
edited by hand after the ranking changed, and a CSV regenerated from a stale
run. Rule 7 is the same argument `analyze-demo` makes for grading
`invocation_spec.json` against `forge_task.yaml` — two exports of one record, so
a disagreement means one was edited alone, and it is invariably the CSV.

## What it cannot catch

**It does not open the profile.** A worklist whose every number is internally
consistent and about a *different capture* passes here. `source.sha256` is what
makes that checkable, and checking it needs the profile, which this validator
was not handed. The mock set contains exactly that trap: two artefacts filed
under the name `kernel_table`, one a real 124-kernel capture and one a 34-row
synthetic seed.

It also does not judge the taxonomy. A kernel in the wrong bucket is a rule-table
gap, not a document defect — measured: aiter's own `cross_device_reduce_1stage`
was 52% of a real profile and fell through to `routable` because the
`AllReduce`/`all_reduce` patterns are written for a name a human wrote.
`todo.md` T4.
