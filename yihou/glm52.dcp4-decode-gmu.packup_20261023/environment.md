# Environment

| Item | Value |
|---|---|
| Cluster | crusoe "spur", MI355X x8 per node (288.0 GiB HBM each, rocm-smi), same-node P4D4 |
| OAI nodes | crsuse2-m2m-227 (job 233906; NODE_FAIL ~01:00Z 10-23), -190 (job 234200), -039/-219/-210 (earlier S1-S3); amdgpu 6.14.14 |
| Burst nodes | crsuse2-m2m-v3-012 (233843), v3-019 (234313); amdgpu 6.16.13; `HSA_ENABLE_SDMA=0` required |
| Image | `infera-sglang:v0522-d395c3b3-1020-mc4340`, ID `sha256:d3d6a9c20ce060961ef13c69a6e281c09e0bd82b4411eb4fe26148d6f9fa854d` |
| sglang | branch `dev/yihou/glm52_dcp_pd` @ `ab7a898b77` (local only; rebuild per the S1-S3 pack-up) + patches 01-03 here, ro-mounted overlays under `/shared_nfs/yihou_glm52/dcp_pd_20261021/src/` |
| PyTorch | 2.9.1+rocm7.2.0 (from the decode log line "Using the Python TreeCore: PyTorch 2.9.1+rocm7.2.0...") |
| Model | `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`, fp8_e4m3 KV, EAGLE 5/1/6, `SGLANG_SIMULATE_ACC_LEN=3.61` |
| Workload | InferenceX `918524ff`, AgentX C80, 1800 s, warm-up 10/lane, dataset `semianalysis_cc_traces_weka_062126`, seed 42 |

Run timeline (UTC): decode-only probes 16:03-16:40Z 10-22; load runs S2g092 17:22Z ... S2m 01:30-02:40Z 10-23
(per-run windows in `results/runs/*/analysis.txt`).

Secrets: crusoe login (ssh alias `spur`), nothing else (model and image on NFS).
