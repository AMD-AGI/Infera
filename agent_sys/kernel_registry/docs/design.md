| | |
|---|---|
| Status | Normative for the kernel registry implementation |
| Version | 2 |
| Updated | 2026-10-28 |
| Summary | Three readable document types and a small CLI organize kernel artifacts in Git. |

## 1. Scope

Use three documents: a kernel description, an environment snapshot and a test
record. Keep domain distinctions inside those documents instead of requiring a
separate object for each concept.

Ownership: `handoff/store.py` already owns storage of workflow content and its
knowledge-store retention. It does not define kernel semantics, optimized version
lineage or benchmark metrics. `validator` invokes checks; `env_mgr` owns workflow
isolation and explicitly excludes the software stack under test. This standalone
domain tool therefore defines a portable content format and its checks. Workflows
may publish it through existing handoffs unchanged. No new lifecycle or database
is introduced and no existing module imports this utility.

## 2. Layout and import graph

```
my-kernels/
  kernels/sparse_mla/kernel.json    original, reference, versions, integration
  environments/rocm72.json         observed stack and hardware
  runs/sparse-iter14.json          one microbenchmark or e2e observation
  payloads/                       source files, docs, small logs and configs
```

`cli.py` uses `registry.py` and `capture.py`; both use the Python standard library.
`templates/` supplies a small working CPU example; the skill lives in
`skills/kernel-artifacts/SKILL.md`. Catalogs are data, not installed Python code.

## 3. Format

[format.md](format.md) is the field reference. JSON uses readable names, not
hash-named records. All document types have `name` and `description`; `kind` is
`kernel`, `environment` or `run`. Unknown descriptive fields are allowed.
Files use `{path, sha256, size}` or `{uri, sha256, size}`; optional `role` describes
purpose. Paths are relative to the catalog and may not escape it through links.

A kernel holds `original`, `reference`, `documentation`, `versions` and
`integration`. A version has `id`, optional `parent`, `files`, `provenance` and
notes. A run names versions through `kernels: {kernel_name: version_id}` and
retains its own `sources` file descriptors, so a later source edit is detectable.
Environment references are readable names. An e2e run includes its integration
files and activation observation. Optional fields improve documentation; missing
reproduction details produce warnings rather than a large admission checklist.

Validation checks structure, references and file integrity. It also reports
missing original/reference/integration documentation, incomplete stack capture,
missing test configuration/evidence and declared historical gaps. It does not
force a completeness claim, require immutable JSON filenames or block importing
old evidence. Git review remains responsible for historical edits.

Metrics carry `name`, `unit`, `direction` (lower/higher), `aggregation`
(ratio/mean_case_ratio/reported), paired baseline/candidate values or case pairs.
History computes ratios and shows separate qualifiers: correctness, confounders,
missing reproduction information and reported-only data. It does not rank across
unrelated runs or promote a version automatically.

## 4. Public surface

```
kernel-registry init CATALOG
kernel-registry validate CATALOG
kernel-registry history CATALOG [--kernel NAME] [--json]
kernel-registry capture-env --name NAME [--repo PATH ...] [--image IMAGE]
                            [--image-digest sha256:...] [--output FILE]
kernel-registry file CATALOG RELATIVE_PATH
```

`init` copies a tiny CPU example into a new directory. Ordinary editing and Git
commands are the write workflow. `file` computes a descriptor without copying.
`capture-env` uses installed-package metadata and bounded read-only probes; it
does not import torch or infer a container digest from its mutable tag. JSON
output supports scripts; history defaults to a short table, with `--json` for
complete records and caveats. Validation returns nonzero only for broken structure,
references or integrity; warnings carry information gaps.

The Python surface is `Registry(root).validate()`, `.history(kernel=None)`,
`file_record(root, relative_path)` and `capture_environment(...)`. No tool
executes commands found in a catalog. Return values are detached snapshots.

The skill has an optional integration workflow for an explicitly requested e2e
run. It uses the existing serving and benchmark tools, records baseline and
candidate observations, and checks actual replacement execution and request
completion. A portable experiment packup can be found by logical name in the
user's configured knowledge repository. The skill does not install a dispatcher.

Before publication, remove private deployment locators from artifacts and names.
When source bytes change during redaction, recompute their descriptors and keep
historical measurements labelled as observations of the pre-redaction sources.
Retain opaque original digests for identity, not private source paths. Such
derivatives need target-environment validation before reuse. Capture labels source
repositories by ordinal and does not substitute local filesystem URLs for absent
remote origins.

## 5. Build versus adopt

Adopt Git, JSON, SHA256 and ordinary files. Build the few kernel-domain relation
checks, history calculation and environment probe. No extra dependencies are
needed. No runtime dispatcher, web service or optimizer is included.

## 6. Test plan

| Spec criterion | Validation |
|---|---|
| 1–2 | Working example; deliberately corrupt payloads, references and paths; incomplete legacy fixture |
| 3–4 | Failed/unknown/confounded runs, lower/higher ratios and mean of case ratios |
| 5 | Missing probe behavior plus real Docker environment capture |
| 6 | Wheel install outside checkout, CLI template workflow and skill validation |
| 7 | Validate imported catalog and compare source digests; CPU-test newly authored naive references |

Tests use CPU data. Importing legacy evidence does not claim GPU verification.

## 7. Deviations from the spec

None.

## 8. New open questions

Future optimizer-specific import adapters can be added when repeated use justifies
them. The core format does not depend on KernelForge or one model family.
