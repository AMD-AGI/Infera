# Environment and provenance

Experiment2026-09-23; hardware/software inventory captured read-only2026-09-24. Later inventory is labeled separately and is not proof of unchanged host state at run time. Run-time effective arguments, GPU snapshots and logs are preserved under environment/run-* and logs/.

## Hardware
| Field | Prefill | Decode |
|---|---|---|
| Host | crsuse2-m2m-138 | crsuse2-m2m-136 |
| IPv4 control/bootstrap |10.245.157.237|10.245.154.168|
| GPUs used |0,1,2,3|0,1,2,3|
| Installed GPUs |8×AMD Instinct MI355X,288GiB each|same|
| CPU |AMD EPYC9575F64-Core;236 visible CPUs|same|
| RAM |~2.7TiB|same|
| Kernel |6.8.0-107-generic|same|
| amdgpu module |6.14.14|same|
| ionic /ionic_rdma |25.08.4.004|same|

RDMA: RoCEv2 global IPv6 GID index1 on ionic0–3, corresponding physical rails pinned per visible GPU. Control endpoints use ens3 IPv4; do not confuse control address family with RDMA GIDs. Full GIDs/driver srcversions in environment/drivers-20260924-node*.txt. GPU-memory write+byte verification passed both directions on all4rails,40.11–42.44GB/s. Host libionic injected from `/lib/x86_64-linux-gnu/libionic.so`, SHA256 `f67c9b4897d5f17e857701ad9c1990765cbe44d67830a47afd44224572072cc3` on both hosts.

## Software
- Exact experiment image local ID: `sha256:fd7220a57b7d3b58efd875c41f7a9ef46b93469581102d96cbeb6f5451e91d35`.
- Tag: `infera-sglang:v0519-yihou-0917-nextnfix-hicache`. NO final registry RepoDigest captured; transfer verified existing image for exact reproduction.
- Vendor ancestor tag: `lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260917`, registry digest `sha256:21c1cc9ab9b703cfe2bf4d62d5c2028654e5f54c810890d9d4b2019d1c31ca32` (archived ancestor evidence).
- SGLang distribution `0.5.19.dev20260917+ga9fb1c3238`; checkout `7ccbf5fd04f7ee23095fc38e49e749d58dc18282` plus image patches and this kit's4file mounts. Version suffix is not checkout SHA.
- AITER checkout `4ad99832823dde2315b361cbd3b54b1c5c12acd5`, ancestor reports dirty. Distribution metadata unavailable; exact image is authoritative.
- Torch distribution metadata `2.9.1+rocm7.2.0.lw.git7e1940d4`; run-time torch.__version__ printed `2.9.1+rocm7.2.0.git7e1940d4`.
- Transformers5.12.1; Triton3.7.0+amd.rocm7.2.0.git89002410; ROCm7.2.0.
- Run worktree branch `dev/pd_opt/glm_5.2_agentx`, HEAD `2e5a2d89f839e69729c56a6cb2786050136d45cb`. Other sessions had uncommitted changes; this experiment used isolated vendored harness and exact image, not the concurrently modified host router.
- Infera installed image build commit is not conclusively identified. Ancestor base reproduction uses5a342acffe10e09729662ff40e81b22a4367fd75 plus archived router/bench patches. Do not substitute current host source for installed image.
- Ancestor Dockerfile pins Mooncake faae8dd4a6309c3ecd47e0721a83b0250d686fa2. Runtime Mooncake distribution version unavailable. Historical Dockerfiles/patches/build verification copied under patches/image-provenance; bit-identical rebuilding is not guaranteed.

## External dependencies and credentials
- Shared model `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`. Original n_routed_experts256, topk8; all attention/KV dimensions retained. New overlay changes routed experts8, ep_size4, synthetic marker, index_sharefalse.
- No dataset download; seed42 random IDs1000–99999. No HF/API token required for this point.
- SSH authorization and permission to use node-local Docker required. Registry credentials only if fetching vendor ancestors; obtain through normal team access, not this kit.
- Large image/model/cache artifacts intentionally excluded. No secret values included.

## Capture files
`capture-20260924-node*.txt`: host inventory, image IDs, later container status. `drivers-*`: provider hash and rails. `software-*`: exact image container metadata. `run-*-server-info.txt`: experiment-time arguments. Raw snapshots can contain unrelated container names but no credentials. Service availability on packaging day is not a renewed benchmark or soak result.
