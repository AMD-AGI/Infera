# GLM-5.2 KernelForge catalog

This catalog publishes redacted GLM-5.2-MXFP4 kernel campaigns and integration
evidence. Historical passes belong to the original bytes; target GPU revalidation
is required before using the published sources and renamed adapters. See [REDACTION.md](REDACTION.md). It is governed by the registry's [spec](../../docs/spec.md) and
[format](../../docs/format.md). Importing or inspecting it does not activate kernels.

From the repository root:

```bash
kernel-registry validate agent_sys/kernel_registry/catalogs/glm52
kernel-registry history agent_sys/kernel_registry/catalogs/glm52 --kernel sparse_mla
kernel-registry history agent_sys/kernel_registry/catalogs/glm52 --kernel dcp_indexer --json
```

| Location | Contents |
|---|---|
| `kernels/<name>/kernel.json` | Original/reference sources, optimized versions, parents, integration and metadata |
| `environments/` | Historical software/hardware snapshots and named gaps |
| `runs/` | Microbenchmarks, real-input failure replay and e2e observations |
| `payload/tasks/` | Redacted baseline sources, historical measurement drivers and workload contracts |
| `payload/campaigns/` | Version source snapshots, retained/rejected results and iteration evidence |
| `payload/overlay/` | Current complete overlay plus earlier adapter snapshots |
| `payload/evidence/` | Redacted logs, configuration and results from the source packups |
| `references/` | New PyTorch semantic references and CPU controls |
| `external_fixtures.json` | Large fixture content identities, sizes and SHA256 values |
| `payload_manifest.json` | Published hashes and separate original content digests |
| [EVIDENCE.md](EVIDENCE.md) | Interpretation of measurements and outstanding reproduction gaps |
| [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | Source attribution and retained licenses |

To rebuild the readable JSON records from retained payloads:

```bash
PYTHONPATH=agent_sys python agent_sys/kernel_registry/catalogs/glm52/import_catalog.py
```

The importer checks every retained payload against its manifest before writing
metadata. It does not execute source payloads, fetch external fixtures or rerun tests.
Ordinary Git review tracks subsequent metadata and implementation changes.

For a microbenchmark replay, stage the selected version under the module name
expected by its preserved `driver.py`, together with `graph_harness.py` and the
required fixtures. Use a separate scratch directory on a sufficiently large data
disk/NFS, and the recorded target stack. The exact task contract and historical
commands are attached to the kernel/run records. Fixture URNs record content identity only; authorized input copies and target
image access must be supplied before replay. Historical shell fragments contain
placeholders and are evidence, not ready-to-run launch scripts.

The complete overlay entry point is `payload/overlay/base`. Activation is via
`PYTHONPATH` and the individual switches recorded in each kernel description;
restart server processes after activation or rollback. The older pd fused-indexer
module path uses the separately retained adapter named in its evidence record.
Keep DCP indexer disabled for serving until its real-input correctness issue is fixed.

New naive references are documented separately in [references/README.md](references/README.md).
Their CPU checks do not certify the historical GPU kernels or full-model output.
