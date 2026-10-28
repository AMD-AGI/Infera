> Redacted historical snapshot. Measurements and correctness statements refer to the original run, not these published bytes. Revalidate the published source and adapters on the target stack before reuse.

# Environment (2026-10-23 04:1xZ - 12:xxZ)

| item | value |
|---|---|
[Private orchestration details omitted.]
[Private access details omitted.]
[Private access details omitted.]
| image | `historical-sglang-image-a`, tar `REDACTED_LOCAL_PATH`, `docker load` -> ID `sha256:5b6565ef3e4e...` on both nodes |
| in image | aiter 2bb549102 (amd-aiter 0.1.25.dev178), flydsl 0.3.4.1, torch 2.9.1+rocm7.2.0, triton 3.7.0+amd.rocm7.2.0.git89002410 |
| sglang | overlay `93034eba91` (`REDACTED_LOCAL_PATH`) mounted on both legs; server-info 0.5.22.dev20261020+geae134d61f |
| model | `REDACTED_LOCAL_PATH` (read-only NFS) |
| harness | shared `REDACTED_LOCAL_PATH` ($H, configs + bench-harness, snapshot in `scripts/shared-harness-snapshot/`) driven through a private copy `REDACTED_LOCAL_PATH` ($KH, in `cluster/kf_e2e_20261023/harness/`) |
| runs | `REDACTED_LOCAL_PATH` (+ `.chain.log`, `.pipeline.log`) |
| aiter JIT cache | `REDACTED_LOCAL_PATH`, seeded from REDACTED_NODE's `437497f5...` (same image content, re-IDed) |
| bench | AgentX (InferenceX 918524ff, aiperf) via bench-harness `agentx_bench.sh`, C56, 3600 s, 10 warm-up requests per lane, sim acceptance 3.61 |

## KF candidates (overlay copies are byte-identical; `overlay/MD5SUMS`)

| kernel | campaign commit | md5 | KF-measured (workset, graph) |
|---|---|---|---|
| sparse MLA (`kf_sparse_mla.py`, full copy of sglang v0.5.21 `triton_sparse_mla.py` + KF edits; one import made 0.5.22-compatible) | `64fbeaa2` (iter 14) | 9dacbc6e... (before the import edit) | 3.165x |
| DSA indexer logits + top-k (`kf_indexer.py`) | `c2cacf5f` (iter 15) | 00665196... | 3.153x |
| MXFP4 MoE (`kf_moe.py`, hooks into aiter fused_moe internals) | `9cd51e60` (iter 14) | 0e1bdc90... | 1.105x |

KernelForge f107746, campaigns in `REDACTED_LOCAL_PATH` (workspace-node / compute-cluster NFS), tags `historical-best-*`.

[Private access details omitted.]
