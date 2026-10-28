# Source attribution

The payload publishes redacted snapshots from historical GLM-5.2 KernelForge
campaigns. `payload_manifest.json` gives each published SHA256 and a separately
named opaque digest of its original bytes. Deployment source locations are omitted.
Required copyright and license notices remain in the corresponding files.

| Source | Material | Preserved license |
|---|---|---|
| [SGLang](https://github.com/sgl-project/sglang) | Sparse MLA, DCP and fused indexer production sources and integration contracts | `payload/licenses/SGLang.LICENSE` (Apache-2.0) |
| [aiter](https://github.com/ROCm/aiter) | Indexer kernels, cache representation and MoE implementation dependencies | `payload/licenses/AITER.LICENSE` (MIT) |
| KernelForge | Optimization campaign metadata, candidates and task harnesses | `payload/licenses/KernelForge.LICENSE` (MIT) |

Indexer source snapshots retain their AMD MIT SPDX and copyright headers. Some
task files combine upstream code with campaign wrappers; upstream ownership
continues to apply to the corresponding portions.

`references/naive.py` is newly authored catalog code based on the operator
contracts and historical task-driver semantics. It is not a historical KF
candidate or a claim that a new GPU correctness test has passed.
