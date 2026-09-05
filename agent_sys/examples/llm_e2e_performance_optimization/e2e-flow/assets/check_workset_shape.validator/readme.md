# `check_workset_shape` — completeness, strong

The workset validates against the merged schema and carries at least three shapes
with runnable correctness and performance entrypoints.

**Shape, not quality.** Whether the entrypoints actually run and whether their
numbers are true is `check_workset_runs`'s question and costs GPU hours. This one
costs seconds and runs first, so a workset missing a file fails before a card is
booked for it.

## What it checks

1. `items/codes/workset.yaml` validates against
   `assets/schemas/workset.schema.json`, which carries the field-level half —
   `min_shapes: 3`, the `status`/`missing_fields` biconditional, the
   `reference`/`baseline` required pair, the abort lists.
2. **Every path the document names exists, is non-empty, and — for an
   entrypoint — is executable.** A `workset.yaml` naming a `run_forge.sh` that
   was never written is the most likely way this artefact is wrong, and a schema
   can only check that the *string* looks like a path. The executable bit is
   checked because `agent/gate.py` refuses a seal for a non-executable `script`
   item *after* the body returns, with a message that does not name the missing
   bit — an AI task once looped to its silent timeout over exactly this.
3. `shapes` corresponds to `workload`, **line for line**: same count, same
   `uuid`, same `axes`, same order. The JSONL is the source of truth and `shapes`
   is its index; a drifted index means `--shape CASE_ID` selects something other
   than what a reader of `workset.yaml` expects. A set comparison would accept a
   re-sorted index, so it is a sequence comparison.
4. No workload line pre-fills `solution` or `evaluation`. Those slots are the
   consumer's; filling them asserts an answer nobody measured.
5. The Definition is a flashinfer-bench Definition: the seven keys, `op_type`
   agreeing with `workset.yaml`, and `reference` and `baseline` both parsing as
   Python and both defining a top-level `run`.
6. **`reference` and `baseline` are not the same source.** Identical means the
   speedup is 1.0 by construction. Conflating them is the single most common way
   a speedup number becomes meaningless and it is invisible in a document where
   both fields are merely present.
7. Exactly one primary shape per operator, and at least one shape with a
   `performance` role.
8. The KernelForge add-on agrees with the base it was generated from: the
   exported `cases.json` case list equals the workload's, in order.
   `forge_driver.py` parses and mentions the four tokens forge-loop's preflight
   requires (`SNR`, `allclose`, `--bench-mode`, `CASE_ID`).
9. `evidence` is present and both reports validate against
   `workset.schema.json#/$defs/{correctness,performance}_report`. Optional in the
   schema — a workset may be shape-checked before it is measured — and required
   here, because a workset reaching a consumer unmeasured is the state M4.3.5 was
   reversed against.
10. No hard-coded host path in `.py` / `.sh` / `.json` / `.jsonl`, and no
    `TODO`/`TBD`/`FIXME` in any `.md` or `.yaml`.

## Rule 10 does not rest on the seal, and `analyze-demo` says it does

That package justifies the rule with *"the seal refuses the whole delivery over
one"*. Measured against the framework: `handoff/store.py:447,494` decline to call
`locality.check` at all — user-ruled 2026-08-31 after the shape heuristic read an
HTTP access-log line as a filesystem path, at a measured 97% false-positive rate.
The sealed `deploy_kit` in the mock set carries `/shared_nfs/...` in five files.

The rule survives on the merit it always had: a script carrying one host's
directory does not run on the next host. So it is scoped to what a reproducer
executes and **skips the environment record**, whose absolute `model_path` is
schema-required — scanning that would reject every conforming workset, which is
how a rule this shape gets deleted rather than fixed.

`workset_io.absolute_paths_in` also strips variable expansions before scanning,
because `"$HERE"/scripts/x.py` reads as absolute otherwise. That is not a
loosening: a `$`-prefixed path is a parameterised one, which is what the rule
asks for.

## What it cannot catch

**It does not run anything.** An entrypoint that is present, executable and
measures the wrong operator passes here. That is `check_workset_runs`.
