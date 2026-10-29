---
name: kernel-artifacts
description: Register, update, inspect, or hand over optimized GPU kernel artifacts and microbenchmark/e2e history in a Git-backed kernel registry. Also use when asked to integrate archived kernels into an e2e environment, run them, and preserve the result using the existing serving and benchmark tools.
---

# Kernel artifacts

Keep the requested artifacts in the user's repository using ordinary files and
Git. The `kernel-registry` CLI helps create a template, check it, and read history;
users may edit JSON directly. Do not introduce a service or another versioning
protocol.

For a request to integrate archived kernels and obtain an e2e run, follow
[Integrate and run](references/integrate-e2e.md). Ordinary registration does not
launch services or benchmarks.

## Read the format

Use `kernel-registry --help` and the module's `docs/format.md` as the field
reference. In an Infera checkout it is at
`agent_sys/kernel_registry/docs/format.md`. With an installed package:

```bash
python -c 'from importlib.resources import files; print(files("kernel_registry").joinpath("docs/format.md").read_text())'
```

For a new catalog, `kernel-registry init CATALOG` creates a small CPU example.
Read its kernel, environment, and run documents before adapting them. For an
existing catalog, retain its naming and layout.

## Register or update

1. Gather original code, naive reference, candidate source, integration instructions,
   documentation, and existing test evidence. Preserve source bytes and provenance.
2. Fill `kernels/NAME/kernel.json`: purpose/constraints, original/reference files,
   optimized versions and parents, author/tool/campaign, integration and rollback.
   Use readable version IDs. Label an adapted overlay source separately if its
   bytes differ from the candidate; identify the actual integrated version.
3. Put reusable stack snapshots in `environments/NAME.json` and each observation in
   `runs/NAME.json`. A run names exact versions and retains source descriptors,
   baseline, environment, commands, workload, correctness, metrics, and evidence.
4. Generate local descriptors with `kernel-registry file CATALOG RELATIVE_PATH`.
   Large external artifacts can use URI, SHA256, and size; do not claim that a
   recorded URI was checked or downloaded unless that happened.

For a new test, capture the environment where it runs:

```bash
kernel-registry capture-env --name NAME --repo /path/to/source --output FILE
```

Repeat `--repo` for relevant sources. Supply `--image` and `--image-digest` from
observed container identity when available. Inspect missing facts and supplement
the workload's relevant stack, driver, build, source-patch, and input information.
An environment captured today cannot fill an unknown historical environment.

## Preserve what the evidence says

- Keep original performance baseline and naive correctness oracle distinct.
- Distinguish synthetic tests, real-input kernel checks, and model-output checks.
  Successful execution alone does not establish numerical correctness.
- Record failures and inconclusive results. List missing historical facts under
  `reproducibility.missing`; import is not a rerun or proof of reproducibility.
- Name confounders such as different nodes, configurations, or inputs. Keep the
  reported aggregation: mean case speedup differs from a ratio of total timings.
- Attribute a bundled e2e result to its kernel set. A fast microbenchmark does
  not establish an e2e improvement or select the deployed version.

## Check and hand over

```bash
kernel-registry validate CATALOG
kernel-registry history CATALOG --kernel NAME
```

History prints a concise table; add `--json` for full records and qualifiers.
Resolve integrity/reference errors; explain remaining information warnings.
Use normal Git diffs and commits to review changes. Keep historical labels and
run names; give changed code or a new observation a new name. Report the catalog,
registered versions, integration choice, validation result, and remaining gaps.
For integration work, also report actual activation, request completion,
correctness scope, and rollback. Registration alone does not authorize deployment.
