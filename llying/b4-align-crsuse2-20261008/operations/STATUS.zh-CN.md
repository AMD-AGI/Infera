# 工作状态

按时间追加。

## 决定（2026-10-08）

- 目标：在本集群（crsuse2）用 1P1D C80 复现 aus 集群的 B4，看性能能否对齐。B4 配置见
  `../router-capacity-campaign-20260928/b4/`，指标见 `../experiment-summary-20261008/`。
- 镜像按 aus 配方重建：0916 公开镜像 + Infera `83e0f6c8` → yihou overlay → tracing →
  p01/p02，与 B4 一致；只额外加 Mooncake v3（B4 关闭 DP rank 亲和，有跨 rank 传输，本集群
  原版 Mooncake 会 transfer failure）。
- `MOONCAKE_DISABLE_HIP_DMABUF` 保持本集群的 0，不照搬 aus 的 1：两个集群网卡驱动不同。
- 节点：Prefill 136，Decode 138。

| 时间 (UTC) | 主机 | 内容 | 状态 / 输出 |
|---|---|---|---|
| 08:51 | 全部 | 节点检查 | 136：他人 `dev_xb` 约 1 小时前重新启动，8 卡 100% 利用、各约 100 GB；137：yihou r007 实验 + `dev_xb`；138：GPU 空闲，07:52 新建了他人的 `dev_xb`（未占 GPU）。不动他人容器，构建改在 138 |
| 08:51 | 138 | `setsid build/build_images.sh`（不带 peer，暂不拷贝） | 日志 `build-images.log` |
| 08:52 | 138 | `build/build_router.sh` 首次 | 失败：bindgen 找不到 libclang（`build-router-fail-libclang.log`）；按 Dockerfile.sglang 改用 ROCm 自带 libclang |
| 08:55 | 138 | `build/build_router.sh` 重跑 | 完成：`artifacts/infera-router-19a6c1d2`，sha256 `4077025d…`（rustc 1.99.0；与 aus 的 `71750540…` 不同属预期）。B4 用到的开关都在；`INFERA_PD_DP_RANK_AFFINITY` 未实现，B4 本就为 false |
| 08:52–08:58 | 138 | 镜像链 | 完成，全部实际构建（0 个 CACHED）。0916 基础镜像 digest `eef7b70e…`；Mooncake `faae8dd4`、libionic 54.0-187-1、DSA/disagg/responses/ROCm HiCache 补丁、NextN、PR #37152（hunk 偏移 -16）、tracing、p01/p02、Mooncake v3 均成功；overlay 校验 `markers=ok sglang=0.5.19.dev20260916+ge7f7447333`，与 aus 一致。最终 `infera-sglang:aus-campaign-radix-20260928-mcdestpin-ibto18` = `e13b5843…`（ID 与 aus 不同属预期：构建时间戳不同）。尚未拷贝到第二台节点 |
| 09:05 | — | 用户决定：先只准备镜像和配置，节点稍后再定 | |
| 09:10–09:50 | 登录节点 | 运行套件 | `runtime/`（aus 脚本原样副本 28 个，见 `MANIFEST.tsv`）；`decode_prefix_diag.py` 由最终镜像的 `decode.py` + aus 补丁生成（fuzz 0 原行号应用）；`config/b4.crsuse2.sh` 与 B4 展开后的 167 个变量逐项比对：102 个相同，其余为本集群绑定项（节点/IP、路径、镜像、DMABUF=0、NODEPORT 用 B4 实际的 25000–32767）；`scripts/run_b4.sh` 的 prepare 在 138 上检查通过（136 无镜像、GPU 被占时按预期拒绝）；InferenceX 已取到 `918524ff` |

| 09:50–10:00 | 138 / 登录节点 | 数据集与客户端基线 | 138 的 `/tmp/inferencex-agentic-liyingli/hf/hub`（1.8 GB）复制到 `runtime/cache/agentx/hf/hub`；Weka `23f152f6…` 快照 8 个文件与 B4 `baseline-validation.json` 的清单 SHA256 全部一致，改为与 B4 相同的离线模式。`runtime/reference/b4` = B4 的 `runtime.env`，只改写 4 个路径键，作为 aus 客户端校验脚本的基线 |

## 待办

- 定第二台节点后：把最终镜像 `docker save | docker load` 到该节点，再按 README 运行。
- 运行后：对比 B4 的 `agentx_conc80.json` 与汇总表指标（Total/Output tok/s/GPU、TTFT
  p50/p90、ITL p50/p90、P miss、D 本地复用、P 排队、D 等待 KV）。
