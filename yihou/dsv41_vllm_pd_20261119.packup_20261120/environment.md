# Environment

## Hardware
- openai cluster (smci355, slurm `Compute-DCPT`, QOS batch): node **smci355-ccs-aus-n10-29**, 8x MI355X (gfx950, 288 GB),
  256 CPUs, 3,023 GiB DRAM, /dev/shm 1.6 TB, 8x ionic 400G RoCE (4 per NUMA). Full facts:
  `results/preflight_n10-29/*/facts.md`. Runs 2026-11-19 10:21Z–13:42Z (job 36507, preempted 13:32Z after the runs' profiling).
- crusoe cluster (spur, `amd-spur-v2`, oai/amd-hyperloom-qos): node crsuse2-m2m-v3-186, job 300344 (from 2026-11-20 05:13Z)
  — E7 PD bring-up, results pending.

## Software
- Serving image: `vllm/vllm-openai-rocm:nightly-60bd6dd2edd175a1c55fd3488f871e4c3d2ef27d@sha256:67a3a0fb79893fc5a0fcff5f9b739ba8fec7e920875a12e0574c2e9f50cd814c`
  (same image as InferenceX CI rows 448240/448242). In-image: vllm 0.34.0rc1.dev571+g60bd6dd2e.rocm100, amd-aiter 0.1.29,
  amd_mori 1.2.3.post1, mooncake-transfer-engine-rocm 0.3.14, nixl-rocm 1.5.0.
- Router: vllm-router 0.1.14 (wheel `vllm_router-0.1.14-cp38-abi3-manylinux_2_28_x86_64.whl`, installed inside the container).
- Client: agentx-harness 1.0.14 in a uv venv (CPython 3.11.16) built on a16-19, package list in the venv's `freeze.txt`;
  InferenceX inferencex-e2e @ db3fc796 (`python -m infx.bench agentic`).
- Dataset: HF `semianalysisai/cc-traces-weka-062126` @ 23f152f6 (traces.jsonl sha256 29b6a19e…), offline cache.
- Repo: infera `dev/yihou/dsv41` based on 962845a65 (kit is self-contained in this packup).

## External paths (not in git)
| what | openai | crusoe |
|---|---|---|
| model (476 G, RO) | /perf_apps/chaox/models/DeepSeek-V4.1-Flash | /shared_nfs/models/DeepSeek-V4.1-Flash (md5 of config + index identical) |
| NFS root (runs, logs, stage, venv, hf_home, wheels) | /perf_apps/yihou/dsv41_vllm_pd_20261119 | /shared_nfs/yihou/dsv41_vllm_pd_20261119 |
| node-local root (/root, /tmp, caches of containers) | /data/yihou/dsv41_vllm_pd_20261119 | /mnt/m2m_nobackup/yihou/dsv41_vllm_pd_20261119 |
| full raw outputs (profile_export.jsonl 88-163 MB each, server/client logs) | /perf_apps/yihou/dsv41_vllm_pd_20261119/runs/<tag>/ | — |

## Secrets needed (values not stored)
- openai: slurm via `ssh openai-login`; node ssh only while holding a job (pam_slurm_adopt). Pool manager session books nodes.
- crusoe: `ssh spur005` login; `spur exec <job> -- …` to reach a held node (no direct ssh).
- Docker Hub pull (anonymous OK for vllm images); HuggingFace not needed at run time (offline cache).
