# Environment

## Time (UTC, 2026-10-22)
| Run | Node | Stage start | Profiling window | Note |
|---|---|---|---|---|
| S1 | crsuse2-m2m-219 (OAI, job 233019) | 12:10:18 | 12:54:40-13:24:40 | |
| S2 | crsuse2-m2m-039 (OAI, job 233048) | 12:14:42 | 12:56:05-13:26:05 | |
| S3 | crsuse2-m2m-210 (OAI, job 233080) | 12:30:47 | 13:16:43-13:46:43 | |
| S3-aux | crsuse2-m2m-v3-028 (burst amd-spur-v2, job 233045) | 12:32:17 | 13:36:30-14:06:30 | containers docker-stopped ~14:33Z |
| (S1 lost) | crsuse2-m2m-151 | 11:26:02 | - | node lost in the 11:32Z mass job end |

## Hardware
- 1 node per run, 8x AMD Instinct MI355X; prefill GPUs 0-3, decode GPUs 4-7; host RAM MemAvailable ~2.6-2.7 TB.
- amdgpu: **6.14.14** on 219/039/210/151; **6.16.13** on v3-028 (burst nodes need `HSA_ENABLE_SDMA=0`).
- KV transfer: Mooncake over HIP IPC (same node); `MC_TE_FILTERS_OVERRIDE=yihou_no_such_rdma_device`,
  `--disaggregation-ib-device {"0":"ionic_0",...,"3":"ionic_3"}`.
- Not captured: per-node `collect_env.sh` snapshot, rocm-smi firmware versions.

## Software
| Item | Value |
|---|---|
| Image | `infera-sglang:v0522-d395c3b3-1020-mc4340`, ID `sha256:d3d6a9c20ce060961ef13c69a6e281c09e0bd82b4411eb4fe26148d6f9fa854d` |
| Image tar sha256 | `7067b016125841acfcdebd2d511b51007133a9a36e8c9432e3271d8ca2523ade` |
| Base image | `lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261020` |
| Infera | commit `d395c3b3` (`origin/feat/sglang-mix-glm5.2-tp4-v0522`, llying's v0522 line) |
| Mooncake | `faae8dd4` + Mooncake#4340 port (`scripts/image/mooncake-pr4340-...diff`) |
| sglang (overlay, ro mount) | `dev/yihou/glm52_dcp_pd` @ `ab7a898b77` |
| etcd | `quay.io/coreos/etcd:v3.5.14` (digest `sha256:42106a001d1be7fd22e48830071ceae0fe760ff2c5df8c76c0e3b586d8a1f0fc`) |
| InferenceX | `918524ff94045b3f091115f1051c22a8588edf2b`, scenario `inferencex-agentx-mvp`, dataset `semianalysis_cc_traces_weka_062126`, 393 entries, seed 42 |

## Model
`/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`, served as `glm5.2-mxfp4`, KV `fp8_e4m3`, EAGLE 5/1/6,
`SGLANG_SIMULATE_ACC_LEN=3.61`.

## Secrets needed (values never recorded)
- Crusoe cluster login (ssh `spur` alias via the user's ssh config; LB host key in a private known_hosts).
- Docker Hub pull for the base image and etcd (anonymous works); HuggingFace not needed (model on NFS).
