# `assets/schemas/` — one JSON Schema per structured artefact

Mission rule G2: *"所有结构化的文档，尽量有自己的 json schema, 该 schema 同时暴露
给 producer & validator"*. This directory is the "同时" — one file, resolved the
same way from both sides by `../lib/schema.py`.

**Deliberately not stubbed.** A schema that validates everything is worse than
no schema: it passes, so nobody looks again. Each file below is written by the
module that owns the artefact, against the **real** sealed sample named beside
it, and is not created until it says something.

| schema | owner | write it against |
|---|---|---|
| `environment.schema.json` | **leader — done** | `cheat_for_mock/stage5-integration/bench_stock/content/items/env/context.json` |
| `deploy_kit.layout.yaml` | m1 | `cheat_for_mock/stage1-deploy/deploy_kit/` (38 files) — M1.1 asks for a *layout* spec, not a JSON Schema |
| `bench_result.schema.json` | m2 | `cheat_for_mock/stage2-profiling/aiperf_baseline/content/items/result/` |
| `kernel_table.schema.json` | m2 | `cheat_for_mock/stage2-profiling/kernel_table/` — the **real** 124-kernel one, not stage 3's 34-row seed |
| `kernel_worklist.schema.json` | m3 | `cheat_for_mock/stage3-analyze/kernel_worklist/` |
| `operator_identity.schema.json` | m3 | `cheat_for_mock/stage3-analyze/operator_identity/` |
| `workset.schema.json` | **m3 + m4 together** | both halves: `stage3-analyze/operator_workset/` and `stage4-kernel-opt/workset/`, plus `../../../../../rank0/` for the flashinfer-bench shape M3.7.5 asks for |
| `kernel_optimization.schema.json` | m4 | `cheat_for_mock/stage4-kernel-opt/kernel_optimization/` |
| `integration_report.schema.json` | m5 | `cheat_for_mock/stage5-integration/integration_report/` — note its verdict is `false` |

## The rule every one of them follows

1. **Write it against a real artefact, then prove both directions**: the real
   document validates, and a document missing what matters is rejected with the
   fields named. `environment.schema.json` was gated that way — the raw
   `context.json` returns eight problems, one per missing field.
2. `$id` is the bare filename, so a sibling may `$ref` it. The registry in
   `../lib/schema.py` loads every `*.schema.json` here.
3. `additionalProperties: false` at the top level, `true` inside sub-objects
   that a site may legitimately extend. A closed schema that rejects a field
   somebody needs gets loosened by whoever needs it; an open one that accepts a
   typo is never noticed.
4. A `structured_text` kind **copies its schema into `items/schema`** at
   production time and its validator checks the copy is byte-identical to this
   directory's (CONTRACT.md §3.4).
