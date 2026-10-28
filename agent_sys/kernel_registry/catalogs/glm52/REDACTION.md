# Published derivatives and historical evidence

This catalog removes deployment-local account names, paths, addresses, host and
service identifiers, private repository locations, local image labels and AI
provider/accounting metadata. Stable neutral node aliases preserve comparisons
between the same node and different nodes. They are not connection targets.
The overlay module is `infera_kernels`; switches use `INFERA_KERNEL_*`.

`payload_manifest.json` hashes the actual published files. Its separately named
`historical_original_sha256` is an opaque digest of the original imported file;
`representation` states whether published bytes changed. No private source locator
or reversal map is included. Historical candidate commit IDs identify historical
versions; their published source snapshots can differ after redaction.

All run correctness and performance outcomes describe historical originals.
They do not validate these published derivatives. Kernel and run records carry
`source_bytes: redacted`, and target GPU revalidation is required before reuse.
The standalone PyTorch references are newly authored semantic references with
CPU controls, not historical GPU test evidence.

Fixtures use `urn:sha256:` identities with sizes and `availability: unavailable`.
No download location or container registry access is supplied. Archived launch
fragments and some driver defaults contain `REDACTED_LOCAL_PATH` placeholders.
Before replay, provide `AITER_CONFIG_FMOE` and `FLYDSL_RUNTIME_CACHE_DIR` for the
MoE driver, and `SGLANG_JIT_CACHE_DIR` for the fused-indexer driver. Use the target
stack's compatible configuration and writable experiment cache directories.
Supply authorized fixtures,
model and container images, stage the renamed overlay, then run operator checks,
activation checks and matched e2e baseline/candidate tests on the target stack.
Save those new results as new runs; do not transfer a historical pass to them.

Technical shapes, metrics, public upstream revisions and required copyright and
license notices are retained. No fresh GPU or serving run was performed as part
of this publication cleanup.
