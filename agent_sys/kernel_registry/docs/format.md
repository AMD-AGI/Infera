# Catalog format

Start with `kernel-registry init my-kernels` and edit the example. Use Git normally.
Three JSON document types are enough; names and directories are human-readable.
Extra descriptive metadata is welcome. Missing historical details cause warnings,
not rejection. Broken links and incorrect file hashes are errors.

## Files

A local file descriptor is `{"path":"payloads/kernel.py","sha256":"<64 hex>","size":123}`.
Generate it with `kernel-registry file CATALOG payloads/kernel.py`.
An external large file uses `uri` instead of `path`; retain its SHA256 and size.
External files are not downloaded by the tool. An optional `role` states purpose.
All local paths resolve inside the catalog. The same file may be reused by records.
Use a neutral storage locator when publishing; `urn:sha256:<digest>` records an
unhosted artifact's identity without exposing its original filesystem location.
It is not a download URL. For redacted snapshots, hashes identify the published
bytes; name any retained original digest separately and mark the representation.

## Kernel: kernels/<name>/kernel.json

| Field | Meaning |
|---|---|
| `kind`, `name`, `description` | `kernel`, unique readable name, purpose |
| `original`, `reference`, `documentation` | Lists of file descriptors; original baseline and naive oracle are separate |
| `reference_status`, `limitations` | `complete` or `incomplete`; known semantic/evidence limitations |
| `contract` | Inputs, outputs, dtypes/layouts, supported shapes and mathematical behavior |
| `versions` | List of optimized versions described below |
| `integration` | Method, adapter files, compatibility, activation and rollback commands |

Each version has `id`, `files` and optional `parent`, `provenance`, `status`, `notes`.
`parent` names another version of the same kernel. Use a new ID when code changes.
`provenance` can name author, AI tool/model, campaign, upstream commit and origin.
`status` is descriptive (for example candidate, integrated, rejected), not a
validation verdict. Integration contains `files`, `method`, `activation`,
`rollback`, `compatibility`; it may name `version` to identify the selected version.

## Environment: environments/<name>.json

| Field | Meaning |
|---|---|
| `kind`, `name`, `description` | `environment`, readable name, capture context |
| `container` | `image` and immutable `digest`; record the full image ID or repository digest |
| `hardware` | `device`, `architecture`, `count`, `driver`; topology/settings may be added |
| `system` | `os`, `kernel`, `python` |
| `packages` | Installed package → version map |
| `repositories` | Source `url`, full `commit`, `dirty`; retain a `patch` file descriptor if dirty |
| `evidence`, `missing`, `provenance` | Optional probe files, explicitly unavailable facts, how captured |

`capture-env` fills what it can observe. Run it in the benchmark environment,
provide container identity from the host, and supplement driver, compiler,
firmware, clocks or framework build settings relevant to the workload. Unknown
values stay null. Avoid copying credentials or arbitrary environment variables.
Captured records still need review before sharing: supplied image names, remote
URLs and attached logs may identify private infrastructure. No local checkout
path is emitted as the fallback for a missing repository origin.

## Run: runs/<name>.json

| Field | Meaning |
|---|---|
| `kind`, `name`, `description`, `timestamp` | `run`, unique observation name, context and ISO timestamp |
| `run_type` | `microbench` or `e2e` |
| `kernels` | Map of kernel name → exact version ID; a bundle may contain several |
| `sources` | Exact version source descriptors retained at test time |
| `baseline` | Original/baseline source and configuration descriptors |
| `environment`, `baseline_environment` | Names from `environments/`; capture each side |
| `workload` | Shapes/dtypes/layouts, seed/distribution, input identity; model/checkpoint identity for e2e |
| `configuration`, `commands` | Test settings and exact invocation, including relevant environment flags |
| `correctness` | `status` passed/failed/not_tested, `scope`, `summary`, `tolerances` |
| `metrics` | Measurements described below |
| `observed_metrics` | Raw unpaired values; preserve them without inventing a baseline or speedup |
| `comparison` | `controlled` boolean and `confounders` list |
| `integration`, `activation` | Exact adapter file descriptors; confirmed/not_confirmed/not_applicable |
| `evidence` | Raw log/result/config file descriptors |
| `reproducibility` | `status` complete/incomplete, `missing` list |
| `limitations`, `provenance` | Caveats, import source, measurement producer |

A complete record requires the actual stack, input identity, commands, configs
and evidence for both sides. For legacy imports, name missing facts and use
`incomplete`; validation provides additional warnings. A full image ID identifies
bits but does not promise the image can still be retrieved.

Each metric has `name`, `unit`, `direction` (`lower` or `higher`) and `aggregation`:

- `ratio`: `baseline` and `candidate` positive finite values; speedup is baseline /
  candidate for latency and candidate / baseline for throughput.
- `mean_case_ratio`: `cases: [{name, baseline, candidate}, ...]`; average the
  individual ratios, not the aggregate timings.
- `reported`: `reported_ratio`; preserve a historical claim when raw pairs are
  unavailable. It remains labelled reported.

History is a ledger, not a leaderboard. Failed, unknown, confounded and incomplete
observations remain visible with qualifiers. Never credit one kernel with a bundle's
entire e2e gain, or translate synthetic/live checks into model-output correctness.
