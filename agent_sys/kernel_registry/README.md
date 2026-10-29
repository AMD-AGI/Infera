# Kernel registry

Keep optimized kernels and their test history in an ordinary Git directory.
Engineers and AI agents use three readable documents: a kernel description,
an environment snapshot, and a test record. Edit them directly and use Git for
versions and review. The CLI creates examples, checks files, and displays history.

The module's requirements are in [docs/spec.md](docs/spec.md).

## Start

From the Infera repository root:

```bash
pip install -e agent_sys
kernel-registry init my-kernels
kernel-registry validate my-kernels
kernel-registry history my-kernels
```

`init` creates a small CPU example in a new directory. Use it as a template in
your own repository; the tool does not require an Infera checkout at runtime.

```text
my-kernels/
  kernels/<name>/kernel.json    original, naive reference, versions, integration
  environments/<name>.json     software stack and hardware
  runs/<name>.json             microbenchmark or e2e observation
  payloads/                   sources, documentation, logs and configuration
```

## Register a kernel or a test

Copy the example documents, give them readable names, and fill in your artifacts.
[docs/format.md](docs/format.md) describes the fields. Keep original code and the
naive correctness reference separate, and name the version actually used by the
integration. A changed implementation gets a new version label.

Generate a file descriptor without copying the file:

```bash
kernel-registry file my-kernels payloads/kernel.py
```

Capture an environment inside the container used for the test:

```bash
kernel-registry capture-env --name benchmark-stack --repo /path/to/source \
  --output my-kernels/environments/benchmark-stack.json
```

Provide `--image IMAGE --image-digest sha256:...` when the host's image identity
is available. Inspect the snapshot and supplement relevant hardware, driver,
compiler, framework, and source-patch details. Unavailable facts remain explicit.

Record the tested versions, baseline, command, inputs, configuration, correctness,
performance, and raw evidence in a run document. Historical gaps belong in
`reproducibility.missing`; incomplete records can still be useful.

```bash
kernel-registry validate my-kernels
kernel-registry history my-kernels --kernel your-kernel
git add my-kernels
git commit -s -m "Record kernel artifacts and test evidence"
```

Validation errors identify broken documents, references, or file integrity.
Warnings identify information to complete. History retains failed, confounded,
reported, and incomplete observations with their qualifiers. It does not select
or deploy a kernel. Add `--json` to `history` for full records, qualifiers, and
observed metrics. Large external files can use URI/checksum descriptors.

## AI skill

Point an agent at [skills/kernel-artifacts/SKILL.md](skills/kernel-artifacts/SKILL.md),
or copy the `kernel-artifacts` folder into that agent's skill directory.
The skill uses this CLI and the same document format; no global configuration
change is required. For requested integration runs, its
[e2e workflow](skills/kernel-artifacts/references/integrate-e2e.md) covers staging,
activation, correctness, rollback, and reusable experiment records.

## Included artifacts

The [GLM-5.2 KernelForge catalog](https://github.com/AMD-AGI/Infera/pull/221)
contains published sources and historical measurements. Catalog data are not bundled
into the installed tool; customers can keep their own catalogs in private repositories.

## Contents

| Path | Purpose |
|---|---|
| `registry.py` | Catalog checks, file descriptors, and test history |
| `capture.py` | Environment snapshot |
| `cli.py` | Command-line interface |
| `templates/` | Small example copied by `init` |
| `skills/kernel-artifacts/` | Artifact registration, e2e integration, and handoff |
| [docs/format.md](docs/format.md) | Document field reference |
| [docs/spec.md](docs/spec.md) | Required behavior and acceptance criteria |
| [docs/design.md](docs/design.md) | Implementation and ownership |
