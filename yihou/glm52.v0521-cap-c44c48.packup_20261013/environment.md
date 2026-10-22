# Environment

Hardware/software beyond this list is identical to the baseline kit
`glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/environment.md` (MI355X ×8, ROCm 7.2, same image recipe).

| item | value |
|---|---|
| nodes used | `crsuse2-m2m-102` (10.245.147.130; spur `amd-oai`/`amd-oai-qos`, job 207495 — TIMEOUT 2026-10-12 20:01), `crsuse2-m2m-137` (10.245.153.247; plain ssh; `/shared_nfs` read-only → node-local `/mnt/m2m_nobackup/yihou_v0521dbg`; since 2026-10-13 ~01:29 used by another user) |
| nodes tried | 174 (burst 209081, cancelled at 20 min), 029 (burst 209186, cancelled at 20 min), 205 (foreign GPU drain pool `crusoe-sgl-q36`) |
| node held at pack-up | `crsuse2-m2m-256` (oai job 208970, RUNNING since 2026-10-12 20:01, idle, image not loaded) |
| image | `infera-sglang:v0521-yihou-1010-mc4340`: 102/174/029 ID `sha256:9d755dd6a7bf2f3705a45dcc7877b9ec3625efb8370ab3b144751799e31300d2`; 137 ID `sha256:e8e2838f4a650584e92f1731cce26d3c7c500724f8befd301ecaac644763e22b` (docker load, same content) |
| image (opt.) | `infera-sglang:v0521-yihou-1010-mc4340prio` `sha256:aacc9aeb…` (102 only; build recipe = baseline kit patch 0005) |
| image tarball | `/shared_nfs/yihou_glm52/v0521cap_20261012/images/infera-sglang.v0521-yihou-1010-mc4340.9d755dd6.yihou.tar.zst` |
| sglang source (for code reading) | `/shared_nfs/yihou_glm52/v0521cap_20261012/src/image-9d755dd6/sglang` (decode.py md5 bd1e5fe7…) |
| model | `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` |
| runtime workspaces | `/shared_nfs/yihou_glm52/v0521dbg_20261010` (harness, runs, caches), `/shared_nfs/yihou_glm52/v0521cap_20261012` (tools, metrics TSVs, image) |
| secrets | none beyond cluster login |
