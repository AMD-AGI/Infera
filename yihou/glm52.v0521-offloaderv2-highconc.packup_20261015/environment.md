# Environment

| node | pool / job | data IP | rounds | kernel / amdgpu driver | image ID | capture |
|---|---|---|---|---|---|---|
| crsuse2-m2m-252 | amd-spur oai 214829 (released 10-15 05:00) | 10.245.149.156 | ofg4c, ofg4 (aborted) | 6.8.0-107-generic / 6.14.14 | 9d755dd6… | env/collect_env.252.20261015.txt |
| crsuse2-m2m-288 | amd-spur oai 215369 (released 10-15 05:00) | 10.245.152.60 | ofg2 | 6.8.0-107-generic / 6.14.14 | 9d755dd6… | env/collect_env.288.20261015.txt |
| crsuse2-m2m-136 | plain ssh, no job (shared; xiaobche's containers present) | 10.245.154.168 | ofg4b, ofg4h, ofg4h2 | see env/collect_env.136.20261014.txt | 9872bbda… (own build) | env/collect_env.136.20261014.txt |
| crsuse2-m2m-v3-020 | **amd-spur-v2** burst 217125 | 10.245.150.105 | ofg3b | **6.8.12-322.crusoe / 6.16.13** | 9d755dd6… | env/collect_env.v3-020.20261015.txt |
| crsuse2-m2m-v3-026 | **amd-spur-v2** burst 217126 | 10.245.152.176 | pf90b, ofg2h, ofg2h2, ofg2h3 | **6.8.12-322.crusoe / 6.16.13** | 9d755dd6… | env/collect_env.v3-026.20261015.txt |

- The `amd-spur-v2` nodes run a different kernel and amdgpu driver (6.16.13) than the amd-spur nodes (6.14.14), so cross-pool score comparisons carry that difference too.
- All nodes: 8 × MI355X, ~2.7 TiB RAM, docker root `/mnt/m2m_nobackup/docker`. Image load logs: `env/prep.<key>.yihou.log`.
- Software: sglang `0.5.21.dev20261008+g11b1fb21de` in `infera-sglang:v0521-yihou-1010-mc4340`; OffloaderV2 + YIHOU-OFFLOAD-ATTRS patch on prefill only; AgentX InferenceX `918524ff…`, `inferencex-agentx-mvp`, 3600 s, warmup 10.
- Harness: `/shared_nfs/yihou_glm52/v0521dbg_20261010/harness` (copy in `scripts/harness/`), not in git. Pack-up repo `/home/yihou/dev/git/infera.glm52.pd` (uncommitted).

## Gaps

- No post-run ECC capture. 136's capture is from 10-14.
- 136 server logs are node-local. ofg4b's were streamed in over ssh at packing; ofg4h's decode log never existed (the decode was never launched).

## Secrets

None. Builder output was grepped for HF_TOKEN / hf_ / ghp_ / AKIA / private keys.
