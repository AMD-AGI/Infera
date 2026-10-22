# Environment

Source: `results/env/collect_env.274.20261014.txt` (captured 2026-10-14 01:57 UTC with `scripts/session/collect_env.yihou.sh`,
i.e. ~7 h **after** the runs; hardware/driver do not change between, but VRAM/containers in that snapshot belong to later work),
`results/rounds/pc48{b,e}/audit/*-server-info.json.gz` (live engine args at bench start), the dryruns, and the sibling kits.

## Hardware — `crsuse2-m2m-274` (all three rounds)

| item | value |
|---|---|
| node / data IP / job | `crsuse2-m2m-274` / `10.245.157.34` (ens3, mlx5_0 200G) / spur `amd-oai` job 211171 (`-A amd-oai -q amd-oai-qos --gres=gpu:8 -c 160`, held from 2026-10-13 05:06 UTC) |
| GPUs | 8× AMD Instinct MI355X (gfx950), 309,220,868,096 B VRAM each; amdgpu driver 6.14.14 |
| CPU / RAM | 2× AMD EPYC 9575F (236 logical CPUs visible), 2.7 TiB RAM, no swap |
| OS / kernel | Ubuntu 24.04.4 LTS, Linux 6.8.0-107-generic |
| RDMA | `ionic_0..7` all Active/LinkUp 400G (Ethernet); harness passes `--disaggregation-ib-device ionic_0-3` on both legs; the engine env sets `MC_TE_FILTERS=yihou_no_such_rdma_device`, `MC_DISABLE_HIP_TRANSPORT=1`, `MOONCAKE_DISABLE_HIP_DMABUF=0` (`results/rounds/*/engine-*.log`). Which Mooncake transport carried the KV was not re-examined in this experiment; `census.txt` records `HIP installed: 4`, `rdma installs: 0` per leg. See the baseline kit for the transport story |
| docker | 29.7.2 |
| ECC | not captured for 274 in this session [gap]; 026 was released for GPU3 UMC correctable 13,198,148 + 2 deferred |

Other captures in `results/env/` (085, 089, 136) are for sibling experiments; nodes 026 and 206 were **not** captured
(026 released, 206 job 211763 ended 2026-10-14 ~01:30, JobHoldMaxRequeue). `env.pc48-206.yihou.sh` was prepared for 206 but pc48
never ran there.

## Software

| item | value |
|---|---|
| image | `infera-sglang:v0521-yihou-1010-mc4340`, ID `sha256:9d755dd6a7bf2f3705a45dcc7877b9ec3625efb8370ab3b144751799e31300d2` on 274 (docker load of the 102 tar; `results/env/imgload.crsuse2-m2m-274.yihou.log`, loaded 2026-10-13 05:10-05:13) |
| image tar | `/shared_nfs/yihou_glm52/v0521cap_20261012/images/infera-sglang.v0521-yihou-1010-mc4340.9d755dd6.yihou.tar.zst` |
| sglang in image | `0.5.21.dev20261008+g11b1fb21de` (server-info `version`); contains PR #40703 (prefill_complete), lacks #43442 / #43407 / #43719 (`analysis/research/pc_upstream.md`) |
| sglang source extracted from the image | `/shared_nfs/yihou_glm52/v0521cap_20261012/src/image-9d755dd6/sglang` (`srt/arg_groups/pd_disaggregation_hook.py` sha256 `ad3a2f7aa22f45167242f05e00cfc2b04d220cf78fbe3a0c58ada2b3ddd935da` = `patches/pd_disaggregation_hook.py.orig`) |
| ROCm in image | 7.2 [second-hand: baseline kit environment.md] |
| model | `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` (mounted ro), served `glm5.2-mxfp4` |
| runtime mounts (both legs) | `mounts.yihou.v2-regbase.tsv`: `kv_events_publisher.yihou-noblock.py`, `forward_mla_rocm.yihou-noabsorbcfg.py`, `conn.yihou-regbase.py` (all in `scripts/harness/mounts-v0521/src/`); pc48b additionally `pd_disaggregation_hook.yihou-attndp.py` (`patches/`) |
| harness | NOT in git: `/shared_nfs/yihou_glm52/v0521dbg_20261010/harness` (copied verbatim to `scripts/harness/`) |
| repo | `/home/yihou/dev/git/infera.glm52.pd` branch `dev/pd_opt/glm_5.2_agentx` @ `a3ba39687e884201839f75a45374cc145d07e7fa` (dirty tree; only this `yihou/` dir is written; the run did not use repo files) |
| AgentX client | dataset `semianalysisai/cc-traces-weka-062126` (393 entries), aiperf; command in `results/*/benchmark_command.txt` |

## Engine shape (identical in all rounds; from server-info + dryruns)

Prefill GPU0-3, decode GPU4-7; each TP4, `attn_dp_size=4` (`--dp-size 4 --enable-dp-attention`, rewritten by sglang), EP1;
triton DSA prefill/decode; fp8_e4m3 KV; EAGLE MTP 5/1/6 with `SGLANG_SIMULATE_ACC_LEN=3.61`; `index_share_for_mtp_iteration=false`;
prefill `--mem-fraction-static 0.88`, `--chunked-prefill-size 32768`, `--max-running-requests 64`, HiCache size 370 GB/rank
(`--hicache-ratio 1.5` still emitted, overridden by size), `--hicache-io-backend kernel --hicache-mem-layout page_first`,
`SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB=512`; decode `--mem-fraction-static 0.96`, `--max-running-requests 96`, no decode HiCache.

| per-rank KV (tokens) | value |
|---|---|
| prefill device pool (mem 0.88) | 2,433,088 |
| decode device pool (mem 0.96) | 2,816,896 (0.95 = 2,762,176) |
| prefill HiCache host pool (370 GB) | 8,235,456 (450 GB = 10,016,064) |

## Secrets

None beyond the spur cluster login. `runtime.env`, engine command logs and container JSONs were grepped for
token/key/password/secret before copying: only `--tokenizer` paths and null `api_key` fields matched.
