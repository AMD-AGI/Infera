# Environment

Time: 2026-10-19 05:2x UTC (task start) → 2026-10-20 01:3x UTC (kit built; g4xc2 still running).

| node | pool / job | data IP | rounds | notes |
|---|---|---|---|---|
| crsuse2-m2m-250 | OAI (`-A amd-oai -q amd-oai-qos -p amd-spur`) job 225298, machine-pool target t-1019210042-7f5d lent to g4tune; started 10-19 13:42Z, MaxWall 24 h | 10.245.158.155 | g4xb5 (A1+A2), g4xm4 (A1+A2), g4xc2 (running) | cache key 250 (prep 10-18 06:51); our 31-h-old ofg3h13-etcd stopped there before g4xb5 |
| crsuse2-m2m-097 | OAI job 225299 (same pool target), 10-19 18:4x Z | 10.245.144.36 | g4xc1 (stopped) | fresh, prep 18:49:50Z; config n17 + `topology.yihou.097.tsv`; xgmi_load microbench 10-20 01:2x Z |
| crsuse2-m2m-v3-017 | amd-spur-v2 burst job 225364 (pool) | 10.245.149.32 | g4xm3 (A1+A2) | fresh key 017, prep 15:12:36Z |
| crsuse2-m2m-v3-021 | burst jobs 225102 / 225141 / 225205 / 225265 (pool, re-granted 4×) | 10.245.145.196 | g4xb1/2/3, g4xm2 (all lost); machine microbenchmarks | foreign container `ixk3-attempt5` present (kfd 0, left running) |
| crsuse2-m2m-v3-032 | burst jobs 225074 / 225242 / 225312 | 10.245.150.194 | g4xm1, g4xb4 (lost) | |
| crsuse2-m2m-v3-026 | burst job 225232 | 10.245.152.176 | none | docker.service failed → unusable (`env/prep.026.yihou.log`) |

- Hardware: 8 × MI355X per node; same-node P4D4 (prefill GPU0-3, decode GPU4-7). `rocm-smi --showtopo` on v3-021: full XGMI mesh, weight 15 everywhere (`results/machine/topo.v3-021.yihou.txt`).
- No `collect_env` capture was taken on any g4tune node. Kernel/driver/firmware for 250/097/v3-017/v3-021 are **not captured**. The closest captures are in the previous kits (`../glm52.v0521-offloaderv2-burst-c56c64.packup_20261019/env/collect_env.v3-020.20261015.txt`: kernel 6.8.12-322.crusoe, amdgpu 6.16.13).
- Image: `infera-sglang:v0521-yihou-1010-mc4340`, ID `sha256:9d755dd6a7bf2f3705a45dcc7877b9ec3625efb8370ab3b144751799e31300d2` (checked at every launch: `img=9d755dd6a7bf`, see `results/launch/`).
- sglang in the image = upstream `sgl-project/sglang` commit `11b1fb21de1c48acee33c6f23a6fc57b1c322341` (`0.5.21.dev20261008+g11b1fb21de`); md5 of `disaggregation/prefill.py` and `decode.py` matched (ledger 10-19 05:2x). Mooncake `faae8dd4` + PR 4340 HIP port (ledger; not re-verified for this kit).
- Engine overlays mounted at run time (not in the image): see `patches/README.md`.
- AgentX bench: InferenceX `918524ff94045b3f091115f1051c22a8588edf2b`, 3600 s, warmup 10/lane, C56 (commit seen in `results/rounds/g4xb5/agentx-c56-A2.log`).
- Model: `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`. Harness: `/shared_nfs/yihou_glm52/v0521dbg_20261010/harness` (not in git; needed files copied to `scripts/harness/`).
- Engine settings from census (`results/rounds/<round>/census.txt`): prefill `max_total_num_tokens=2938752`, per-DP `chunked_prefill_size=8192` (g4xc1: 16384), `max_prefill_tokens=16384`, HiCache host pool 10,016,064 tok / 450 GB per DP; decode `max_total_num_tokens=2762176`, `max_running_requests=24` per DP.
- Prep logs for the nodes: `env/prep.<key>.yihou.log`.

## Secrets

None in the kit. Needed: spur login (ssh `spur00{5,6,7}`), the `amd-oai` account / `amd-oai-qos` (or `amd-burst` / `amd-burst-qos`) via the machine-pool manager.
