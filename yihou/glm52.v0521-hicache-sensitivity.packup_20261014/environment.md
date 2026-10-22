# Environment

## Run of record: hc50 on crsuse2-m2m-136 (captured 2026-10-14 01:57:54 UTC, `env/collect_env.136.20261014.txt`)

| item | value |
|---|---|
| host | `crsuse2-m2m-136`, data IP 10.245.154.168 (`ens3`, mlx5_0), plain ssh (no slurm job), `/shared_nfs` mounted **read-only** |
| OS / kernel | Ubuntu 24.04.4 LTS, Linux 6.8.0-107-generic, uptime 47 d at capture |
| CPU / RAM | AMD EPYC 9575F 64-Core, 2 sockets, `lscpu` CPU(s) 236; 2.7 TiB RAM, no swap |
| GPU | 8 x AMD Instinct MI355X (gfx950), 309,220,868,096 B VRAM each; amdgpu driver 6.14.14 |
| RDMA | ionic_0..7 Active, 400 Gb/s, Ethernet (RoCE), netdevs enP2p0s9-12 / enP3p0s9-12; mlx5_0 200 Gb/s = `ens3` |
| KV transfer | same-node: mooncake with `MC_TE_FILTERS=yihou_no_such_rdma_device` (no RDMA device), census "rdma installs 0", "HIP installed 4" per leg -> GPU IPC path |
| docker | 29.6.1 |
| image | `infera-sglang:v0521-yihou-1010-mc4340`, ID `sha256:9872bbdabfcb8f6676b04378d8ea75c922dfc8db6a8c5c87acd801e205b3b575` (136's own build; both legs, `IMAGE_IDS` in `results/hc50/A2-C40/runtime.env`) |
| Mooncake engine.so | sha256 `3c96489a0ca4c8662a5f6512c1d9d44653835afc3d7cfaa9082b70f575bb289a`, recorded as identical on 102 and 136 in `config.yihou.mc4340n.sh:17-19` (not re-verified for this packup) |
| containers at capture | hc50 stack still up (`glm52-v0521-yihou-136-hc50-{router,decode-0,prefill-0,etcd}`, 9-10 h), foreign `dev_xb` (lmsysorg/sglang:v0.5.20-rocm724-mi35x, up 5 d), Crusoe agents (vector, amd-log-collector, amd-exporter) |
| node-local RW root | `/mnt/m2m_nobackup/yihou_v0521dbg` (caches, tmpfs, server logs, aiperf venv) |
| AgentX client | aiperf in `/mnt/m2m_nobackup/yihou_v0521dbg/cache/136/agentx/aiperf/venv`, InferenceX commit `918524ff94045b3f091115f1051c22a8588edf2b` (clean), scenario `inferencex-agentx-mvp`, dataset `semianalysis_cc_traces_weka_062126` (393 entries), seed 42 |

## Reference e2g on crsuse2-m2m-102 (not captured in this session)

102 (10.245.147.130, spur oai job 207495, ended 2026-10-12) ran image ID `sha256:9d755dd6a7bf2f3705a45dcc7877b9ec3625efb8370ab3b144751799e31300d2`
with caches on NFS (`$WS/cache/102`). Its hardware/driver capture is in the baseline kit
`/home/yihou/dev/git/infera.glm52.view/glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/environment.md`.
**Gap:** no driver/kernel snapshot of 102 at e2g time (2026-10-11) is in this packup.

Other session nodes (captured 2026-10-14 01:57, not used by hc50): 085/274/089 in
`/shared_nfs/yihou_glm52/v0521cap2_20261013/env/collect_env.{085,274,089}.20261014.txt`. 026 and 206 were never
captured (026 released for GPU3 HBM ECC 13,198,148 correctable + 2 deferred; 206 job ended 2026-10-14 ~01:30).

## Software (both runs)

| item | value |
|---|---|
| sglang | 0.5.21.dev20261008+g11b1fb21de (live `server_info.version` of both hc50 and e2g prefill, `results/hc50/round/audit/` and `results/reference-e2g/audit/`), source copy for reading: `/shared_nfs/yihou_glm52/v0521cap_20261012/src/image-9d755dd6/sglang` |
| runtime mounts | `mounts.yihou.v2-regbase.tsv`: kv_events_publisher (noblock), forward_mla_rocm (noabsorbcfg), mooncake conn (regbase diag) — `patches/` |
| model | `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` (read-only mount) |
| topology | same-node 1P1D P4D4: prefill GPU0-3, decode GPU4-7, TP4 DP4 DP-attention each, EP1; triton DSA prefill+decode; fp8_e4m3 KV |
| prefill | mem 0.85, chunked 32768, max_running 64, HiCache write_through / kernel IO / page_first, `--hicache-ratio 1.5` + `--hicache-size 50` (hc50) or 450 (e2g) — size overrides ratio (census: 1,112,896 tok = 50.00 GB per rank), `SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB=512` |
| decode | mem 0.95 (2,762,176 tok/rank), max_running 64, EAGLE 5/1/6, `SGLANG_SIMULATE_ACC_LEN=3.61` (correctness waived), `--disable-custom-all-reduce` |
| router | infera rust router, kv-aware, prefill overlap 20.0 / decode 2.0, `INFERA_PD_DP_RANK_AFFINITY=true` |
| harness git | the harness is NOT in a git repo (`/shared_nfs/yihou_glm52/v0521dbg_20261010/harness`; verbatim copy in `scripts/harness/`). Pack-up repo: `/home/yihou/dev/git/infera.glm52.pd` branch `dev/pd_opt/glm_5.2_agentx` @ `a3ba39687e884201839f75a45374cc145d07e7fa` (dirty) |

## Secrets

None used beyond cluster SSH (workstation -> `spur006` -> `crsuse2-m2m-136`). `runtime.env`, container JSONs and
server-info JSONs were grepped for token/key/password/secret before copying: only `api_key: null` /
`admin_api_key: null` server args and a `SandboxKey` docker netns path matched.
