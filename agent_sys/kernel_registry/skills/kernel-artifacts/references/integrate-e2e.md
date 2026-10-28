# Integrate archived kernels and run e2e

Use this procedure when the user requests integration or a run. Use the project's
existing launcher, benchmark, and correctness checks; the registry records their
artifacts and results. It is not another serving harness.

## Choose a compatible version

Read the kernel's integration instructions and the supporting e2e run, environment,
and limitations. Match hardware, framework/source revision, module paths, model
format, parallelism, and the kernel's shape/dtype guards. Select the documented
integrated version rather than assuming the fastest candidate was integrated.
Exclude versions with unresolved correctness failures from ordinary serving runs.

Identify the existing start/stop, health, request, benchmark, and correctness
commands. Establish the intended success claim: replacement executed, requests
completed, and the applicable correctness check passed. State the scope of that
check; performance improvement is a separate measurement.

## Verify, then stage

Run `kernel-registry validate CATALOG` before copying or adapting code. Check the
selected source and adapter descriptors against their actual bytes. Obtain any
external inputs needed for this run and check their recorded hashes too.

A published redacted derivative may differ from the historical tested source.
Keep its current SHA256 and the retained historical original SHA256 distinct;
validation of the derivative does not prove identity with the original. Verify
the original hash before adaptation when those bytes are available. If unavailable,
record that gap and treat the published derivative as a new target to validate.

Stage the complete adapter and selected sources in a separate experiment directory.
Parameterize machine-specific locations and endpoints, for example `CATALOG`,
`STAGE`, `MODEL_PATH`, and `BASE_URL`; keep their actual bindings out of published
materials. Mount the stage into the target container. For an overlay, put its
directory first on `PYTHONPATH` and use the recorded per-kernel switches.

If target APIs differ, make the smallest adapter change in the stage, retain the
diff, and assign it a new version label and checksum. Preserve the archived input.
An import success only proves that imports worked.

## Run a baseline and the integration

Capture the target stack and exact launcher/configuration. Run the original path
first, then the selected replacement under matched workload and resource settings.
Restart the relevant processes after changing overlay paths or switches. Keep
unrelated configuration fixed and retain both runs' logs and request results.

For the integration, establish all of the following from the run:

- The service became ready and the benchmark's intended requests completed;
  record completion/error counts and exit status.
- The replacement actually executed on eligible inputs. A hook-installed message
  or an enabled switch is insufficient when shape guards can route to fallback.
  Use an execution counter, profiler trace, or an actual replacement-side check.
- The applicable numerical/output check passed with its recorded tolerance and
  input scope. Read its result, including caught errors; a process exit code cannot
  stand in for the check.

Allow the workload's normal build and graph-capture time; inspect progress rather
than declaring a hang from quiet logs. During long runs, assign monitoring and
check health, request progress, errors, and storage periodically.

If execution or correctness fails, retain the evidence, disable the replacement,
restart the original path, and confirm baseline service recovery. Use that
baseline as the differential reference before another narrowly scoped attempt.
Do not relabel failure as success by weakening the test.

CPU smoke tests and synthetic acceptance demonstrate only their measured scope.
A full-model correctness claim requires the corresponding model-output evaluation;
neither live kernel comparisons nor a completed synthetic-acceptance benchmark
establishes it. Compare performance only after explaining any confounders.

## Record and share

Add a run document with exact kernel/adapter versions, both environments, commands,
workload/input identity, baseline, activation evidence, completion, correctness,
measurements, and remaining gaps. Include executable reproduction and rollback
instructions using the parameterized locations. Validate the catalog and inspect
`kernel-registry history CATALOG --json` before committing.

When a reusable experiment packup is requested, use the existing handoff workflow
from [Infera PR208](https://github.com/AMD-AGI/Infera/pull/208), if available, and
resolve the user's configured knowledge repository. Do not embed a private repo
URL in public artifacts. Search for the logical example `glm52-dpa-kernel-e2e`:
it illustrates archived DPA runtime/benchmark success, with simulated acceptance,
without claiming a new GPU rerun or model-quality validation. Read its evidence
and limitations before adapting it to another environment.

Publish portable commands and evidence only. Inspect file contents, names, and
generated metadata for hostnames, personal paths, private endpoints, credentials,
and source-host fields. Keep redaction provenance and original checksums without
publishing private redaction mappings. Use the existing repository's review path;
call an informal packup unsealed unless it actually passed the handoff validators.
PR208's backup path can generate source-host and absolute-path metadata. Inspect
the actual outgoing metadata before using it; for an informal packup, a normal
Git review branch with sanitized `META.astk.json` avoids publishing those fields.
