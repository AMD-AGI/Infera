# Environment

| node | pool / job | data IP | rounds scored | image ID | capture |
|---|---|---|---|---|---|
| crsuse2-m2m-v3-008 | amd-spur-v2 burst 219668 (10-16) | — (see `results/rounds/ofg2h5/`) | ofg2h5 A1 | 9d755dd6… | `env/prep.008.yihou.log` |
| crsuse2-m2m-v3-010 | amd-spur-v2 burst 219127 (10-16) | — | ofg3h2 A1, A2 | 9d755dd6… | `env/prep.010.yihou.log` |
| crsuse2-m2m-v3-006 | amd-spur-v2 burst 220965 (10-17), 222453 (10-18 05:46:55 → 10-19 01:51:13) | 10.245.144.8 | ofg2h17 A1; ofg2h21, ofg3h14, ofg2h22, ofg4h5 (A1+A2); ofg4h4 (unofficial) | 9d755dd6… (checked at every launch: `img=9d755dd6a7bf`) | `env/prep.006.yihou.log` |

- Lost-round nodes (no score): v3-001/007/012/019/025/027/031/033 and amd-spur m2m-011/062/179/200/219/228/232/241/250/274. Prep logs for all in `env/`.
- `amd-spur-v2` nodes: 8 × MI355X; the one captured v3 node (v3-020, previous kit, `env/collect_env.v3-020.20261015.txt`) ran kernel 6.8.12-322.crusoe / amdgpu 6.16.13. Docker root `/mnt/m2m_nobackup/docker`.
- Software: `infera-sglang:v0521-yihou-1010-mc4340` (ID `sha256:9d755dd6a7bf2f3705a45dcc7877b9ec3625efb8370ab3b144751799e31300d2`), sglang `0.5.21.dev20261008+g11b1fb21de`; OffloaderV2 + YIHOU-OFFLOAD-ATTRS patch on prefill only (`patches/`); AgentX InferenceX `918524ff94045b3f091115f1051c22a8588edf2b` (clean), 3600 s, warmup 10/lane, `SIMULATE_ACC_LEN=3.61`.
- Model: `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`.
- Harness: `/shared_nfs/yihou_glm52/v0521dbg_20261010/harness` (verbatim copy in `scripts/harness/`), not in git.

## Gaps

- No `collect_env` capture on v3-006/008/010 (nodes are gone). Kernel/driver for those nodes is assumed equal to v3-020's, not measured.
- No post-run ECC capture. burstkeeper's ECC check at allocation reported 0/0 for v3-006 (10-18).

## Secrets

None in the kit. Needed to reproduce: spur login (ssh to `spur00{5,6,7}`), the `amd-burst` account / `amd-burst-qos`. Builder output grepped for HF_TOKEN / hf_ / ghp_ / AKIA / private keys.
