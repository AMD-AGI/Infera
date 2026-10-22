# Environment (2026-10-21)

| item | value |
|---|---|
| nodes, same-node | crsuse2-m2m-097 (idxf1a, 10-20), crsuse2-m2m-268 (OAI job 227779), crsuse2-m2m-054 (OAI job 228279); spur slurm, `-A amd-oai -q amd-oai-qos -p amd-spur`, held by the machine pool |
| nodes, cross-node | crsuse2-m2m-135 (prefill, IP 10.245.148.209) and crsuse2-m2m-136 (decode, IP 10.245.154.168), plain ssh |
| GPU | 8 x AMD MI355X per node; same-node: prefill GPU0-3, decode GPU4-7; cross-node: GPU0-3 on each node |
| kernel / amdgpu | 6.8.0-107-generic / 6.14.14 on 135, 136 and 268 |
| host RAM | 2751 GB (135, 136, 268) |
| RDMA | 8 x ionic (Pensando, libionic.so.1.0.54.0-149) + mlx5_0; rails physically isolated (ionic_i reaches only ionic_i); MC_GID_INDEX=1 |
| image | `infera-sglang:v0521-yihou-1010-mc4340`; ID 9d755dd6a7bf... on slurm nodes, e8e2838f4a65... after docker load on 135/136; sglang 0.5.21.dev (upstream 11b1fb21de) + local diff; Mooncake faae8dd4 + PR #4340 port |
| model | `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` (real 256 experts) |
| bench | AgentX via `bench-harness/agentx_bench.sh`, 3600 s, 10 warm-up requests per lane, client on the prefill node |
| repo | `AMD-AGI/Infera` branch `dev/pd_opt/glm_5.2_agentx` (local `~/dev/git.16-19/infera.glm52.pd`, HEAD a3ba3968); harness on NFS is not in git |
| raw env snapshots | `env/collect_env.crsuse2-m2m-135.txt`, `env/collect_env.crsuse2-m2m-136.txt` |

Secrets / access needed (no values recorded): spur login (ssh to spur006/007/005), membership of the amd-oai account for OAI
jobs, GitHub token of `dorado269` for the upstream PRs (pushed over ssh), docker on the nodes as user yihou.
