# Environment (2026-10-22 03:30Z - 2026-10-23 03:00Z)

| item | value |
|---|---|
| nodes | crsuse2-m2m-135 prefill GPU0-3 (IP 10.245.148.209), crsuse2-m2m-136 decode GPU0-3 (IP 10.245.154.168), plain ssh; dcps1 only: 136 alone, decode GPU4-7 |
| GPU / kernel / amdgpu | 8 x MI355X per node; 6.8.0-107-generic / 6.14.14 (snapshots: `../glm52.v0521-baseline-xnode-g3g4.packup_20261022/env/`) |
| RDMA | 8 x ionic + mlx5_0; rails isolated (ionic_i reaches only ionic_i); MC_GID_INDEX=1 |
| image dcpx1-2 | `infera-sglang:v0522-1020-yihou-dcp-mc4340`, ID sha256:437497f5b1f7... |
| image dcpx3-6, dcps1 | `infera-sglang:v0522-1020-yihou-dcp-mc4340-destpin`, ID sha256:1b78f7ccce06... (built 10-22 05:0xZ, `imgbuild/`) |
| sglang | overlay 93034eba91 on both legs; decode server-info version `0.5.22.dev20261020+geae134d61f` |
| Mooncake | faae8dd4 + PR #4340 port (+ dest-local-rail, ib-timeout, dest-pinned-hca in -destpin) |
| model | `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` |
| bench | AgentX `bench-harness/agentx_bench.sh`, 3600 s, C56, 10 warm-up requests per lane |
| decode (dcpx6, server-info) | TP4 EP4 DP1, dcp 4, no DP attention, mem 0.85, max_running 128, graph max bs 128, max_total_num_tokens 8,869,120, EAGLE 5 steps, tok/detok 4/4, `index_share_for_mtp_iteration=true`, ib devices ionic_0..3 |
| prefill (dcpx6) | g3x1 prefill: TP4/DP4 DPA, OffloaderV2 G3K1s1, HiCache 260 GB/rank, mem 0.90, chunk 32768, ib devices ionic_0..3 (destlocal) |
| repo | AMD-AGI/Infera `dev/pd_opt/glm_5.2_agentx`; NFS harness not in git (copies in `scripts/`) |

Secrets / access (no values): spur login, docker on 135/136 as yihou, GitHub push rights for this repo.

Rounds of this pack-up (dcpx7..12, dpax1) use the `-destpin` image (ID 1b78f7cc) on both nodes, the same nodes and the
same overlay; per-round decode/prefill values are in `scripts/envs/` and `results/dryruns/` (the decode row above is
dcpx6's). dpax1 decode: TP4/DP4 DPA, mem 0.95, max-running 96, graph max bs 64.
