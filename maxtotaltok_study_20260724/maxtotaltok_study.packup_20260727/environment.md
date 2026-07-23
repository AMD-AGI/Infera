# Environment — DSv4-Pro 1P1D KV-cache 真实用量 + max-total-tokens 调优

Ran: 2026-07-24（max-total-tokens 扫描）+ 2026-07-26（prefill/decode 真实用量 + 优雅回退）。
两轮实验，跨节点 1P1D PD（prefill ↔ decode over mooncake RDMA）。

## Hardware

| 项 | chi2832 (prefill, 2026-07-26 轮) / chi2879 (prefill, 2026-07-24 轮) | chi2878 (decode, 两轮) |
|----|----|----|
| GPU | 8× AMD Instinct MI355X (gfx950), 288 GiB/卡 | 同左 |
| VRAM total | 309,220,868,096 B = 287.98 GiB/卡 | 同左 |
| CPU | AMD EPYC 9575F 64-Core | 同左 |
| RAM | ~3023 GiB | 同左 |
| kernel | 6.8.0-107-generic (chi2832) | 6.8.0-134-generic |
| ionic NIC | 8 active (RoCEv2) | 8 active |
| data-plane IP | chi2832=10.2.122.79 / chi2879=10.2.122.10 | 10.2.122.3 |
| RDMA rail | ionic_0..7, GID index 1 (ULA fd93:...RoCEv2). MVP `ib_write_bw -d ionic_0 -x 1` = **338-339 Gb/s** 健康同 rail | 同左 |

> 起测前 8 卡 VRAM ≈ 0.3 GB/卡（干净）。所有节点直接 ssh via 跳板 root@149.28.124.225 ProxyJump。
> **2026-07-24 轮用 chi2879 作 prefill；2026-07-26 轮改用 chi2832 作 prefill**（chi2878 decode 两轮不变）。

## Software

| 组件 | 值 |
|------|----|
| docker 镜像 | `lmsysorg/sglang-rocm:v0.5.13-rocm720-mi35x-20260612` |
| 镜像 base digest | `sha256:9365640987dbf6db2df79648ff84aa9a214ab1c78cca539433480880a7c3a95b` |
| 镜像 local Id | `sha256:3b01fdb46a95134d54dc3f974c31532da8f625790063e931f4cc926b47dd91f1` |
| **sglang** | **0.5.13**（editable install `/sgl-workspace/sglang`, git commit `50815d54a7b6502342aa037cf462cb1677190a82`）|
| sgl_kernel | 0.4.3 |
| torch | 2.9.1+rocm7.2.0.git7e1940d4 |
| kv-cache-dtype | fp8_e4m3（默认）|
| 每 full-token KV 成本 | **32,781.44 B**（0.5.13 DSv4 实测复合值，含 swa/c4/c128/state 加权）|

> **为何点名 0.5.13**：0.5.13 `_apply_token_constraints`（model_runner_kv_cache_mixin.py:851）=
> `token_capacity = min(profiled, user_limit)` —— `--max-total-tokens` 直接干净地下砍 full_token 池
> （swa/c4/c128 按比例缩）。0.5.15 是多池语义更复杂。

## Repo

- repo：`infera.rdma`，分支 `yihou.dev.rdma`，commit `6408892e3b9758ab4473a7581bc3ce963b521645`。
- packup 所在：`maxtotaltok_study_20260724/maxtotaltok_study.packup_20260727/`。

## 固定并行配置（两轮共用，只扫 max-total-tokens / max-running / conc）

`--tp-size 8 --dp 8 --enable-dp-attention --ep-size 8`，`--attention-backend dsv4`，
`--page-size 256`，`--context-length 9472`，`--disable-radix-cache`，`--swa-full-tokens-ratio 0.15`,
`--cuda-graph-max-bs 512`，`--chunked-prefill-size 163840 --max-prefill-tokens 163840`,
`--disaggregation-transfer-backend mooncake`。prefill gmu 0.85 / decode gmu 0.90。

## 关键 env（launch_leg.sh 头部 verbatim）

R4 perf env set（SGLANG_USE_AITER=1 + fused-compress 全家桶 + `SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton`）
+ mooncake RDMA：`MC_GID_INDEX=1 MC_DISABLE_HIP_TRANSPORT=1`，`SGLANG_HOST_IP=<data-plane>`,
`SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1800`。完整见 `scripts/launch_leg.sh`。
