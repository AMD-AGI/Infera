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

| 09:43 | 全部 | 节点检查 | 136、138 GPU 空闲（各有他人未用 GPU 的 `dev_xb`，不动）；137 被 yihou r007 占用；135 半占。用户决定现在用 136(P)+138(D) |
| 09:52–09:56 | 138→136 | 最终镜像 `docker save \| docker load` | 两端 ID 均为 `e13b5843…`（`image-transfer-138-to-136.log`） |
| 09:56 | 136 | `B4_PREFILL_NODE=crsuse2-m2m-136 B4_DECODE_NODE=crsuse2-m2m-138 setsid scripts/run_b4.sh all`（PID 3836930） | `runtime/runs/b4-crsuse2-136-138-20261008T0956Z/`，日志 `run-136-138.log`；PREPARED 通过 |
| 10:02 / 10:04 | 138 / 136 | D（真实接受率）就绪；P 服务进程起来后做 PD 预热（首次 Triton/AITER JIT 编译，健康检查暂 503） | |
| 10:08 | 136 | `setsid scripts/stop_after.sh 3836931`（PID 3859293） | 测量完成即删除本实验容器；失败或编排进程退出则保留 1 小时后删除。日志 `run-136-138-stop.log` |

| 10:20–10:21 | 136/138 | 第一轮 gate | P 10:20 就绪（首次 JIT 约 24 分钟）。6 题中第 6 题答错：问记录 396（应为 `Z3622`），答了记录 397 的 `Z5040`；temperature 0，P DP5→D DP5 同 rank、冷请求、无前缀复用。`FAILED_GATE_SERVICES_PRESERVED`，C80 未开始；11:23 `stop_after.sh` 删除容器 |
| 10-09 02:30 | 136 | 第二轮：`B4_GATE_SOFT=1 B4_ANSWER_CHECK=1 run_b4.sh all`（编排 PID 420005） | `runs/b4-crsuse2-136-138-20261009T0230Z/`，日志 `run2-136-138.log`。P 02:46 就绪（JIT 已缓存，约 16 分钟） |
| 02:49 | 136/138 | 第二轮 gate | 通过：16/16，覆盖 P/D 各 8 个 rank（这 16 个请求都是同 rank 配对）。第一轮的答错不是系统性的 |
| 02:50 | 136/138 | `answer_check.py` 第一个请求 | HTTP 500：P DP3→D DP4 跨 rank 传输 `cqe with error 12`，两端 `KVTransferError`。`FAILED_ANSWERS_SERVICES_PRESERVED` |
| 03:09 | 136/138 | `run_b4.sh stop` | 已删除本实验容器（`stop_after.sh` 约 03:51 醒来时为空操作） |

| 03:17 | 136 | 第三轮：`B4_GATE_SOFT=1 B4_ANSWER_CHECK=1 setsid scripts/run_unattended.sh`（共享 8 网卡；等显存释放后启动，结束后自动 `stop_after.sh`） | `runs/b4-crsuse2-136-138-20261009T0317Z/`，日志 `run3-136-138.log`；编排 PID 481341 |
| 03:25–03:28 | 136/138 | 第三轮 gate + `answer_check.py` | 两端 `disaggregation_ib_device` 均为共享 8 网卡。gate 16/16（覆盖 P/D 各 8 rank）。答案检查 136 个请求 0 个传输错误；128 个样本全是跨 rank（8 组 P/D 组合），错 1；第二轮 64/64 对且全部命中 D 前缀。两个答错都是同一道题（种子 6 文档、记录 396 → 答成 397 的 `Z5040`）：原题冷重放 4 次错 1（P0→D0），热重放 4/4 对，新文档中同内容的第 5 份第一轮错（P1→D5）。其余 127 个问题全对 |

| 03:33 | 136/138 | 第三轮 switch + preflight + C80 首次尝试 | D 切到 3.61、router 重启；preflight 通过，KV 容量与 B4 偏差 P 0.0081%、D 0.0085%。C80 6 秒后失败：`runtime/` 漏复制了 `validate_chunk8k_runtime.py`（`validate_and_pin_client.py` 调用它），客户端未发出请求；半截产物改名 `c80-attempt1-missing-validator/` |
| 03:34 | 136 | 修正 | 停掉两个处于宽限期的 `stop_after.sh`（第二轮那个按前缀删容器，约 03:51 会删掉第三轮服务）；补上该脚本（`MANIFEST.tsv` 29 个文件）；用上次的 `runtime.env` 预演校验：通过，与 B4 只差允许的路径/主机键（另：主机内存 2752 GB，aus 3023 GB） |
| 03:34 | 136 | `B4_RUN_ID=…0317Z setsid scripts/run_unattended.sh preflight measure` | preflight 再次通过（重新清空缓存），03:34:54 开始 C80；日志 `run3-measure-136-138.log` |

| 03:36–04:56 | 136/138 | C80 | warmup 884 个请求后正式窗口 3640 秒：完成 10,644、取消 11、错误 0；两端无传输失败 |
| 05:02 | 136/138 | `MEASUREMENT_COMPLETE` | `stop_after.sh` 05:03 删除本实验容器 |
| 05:05 | 登录节点 | `analyze.py`、`analyze_decode_prefix.py`、`scripts/compare_b4.py` | 结果见 `../RESULT.zh-CN.md`：吞吐对齐在 1% 以内，TTFT/ITL p90 高 6–8%；数据复制到 `results/b4-crsuse2-136-138-20261009T0317Z/` |

### 第三轮答案检查的结论

- 跨 rank 传输在共享 8 网卡下正常（128/128 无错误）。
- 唯一出错的是同一道题，在临界点上非确定地读错相邻行；同 rank、跨 rank 都出现，与传输和
  D radix 无关。第一轮 gate 的第 6 题就是这道题（gate 的 index 5 turn 0）。三轮合计这道题
  11 次中错 3 次；aus 的 B4/RB gate 各问过一次，都答对，样本太少，不能据此判断两集群有差异。

### 第二轮的根因：B4 的每 rank 单网卡配置在 crsuse2 上跨 rank 不可达

- B4（aus）给每个 DP rank 只配一张网卡：`--disaggregation-ib-device {"0":"ionic_0",…}`，两端
  server-info 均如此。
- crsuse2 的 rail 隔离，`ionic_k` 只通 `ionic_k`（`../cross-rank-rca-20260921/REPORT.zh-CN.md`
  第 276、376 行）。P3 只有 ionic_3、D4 只有 ionic_4，路径不存在；v3 的目的端固定 rail 已生效
  （容器内 `MC_ENABLE_DEST_LOCAL_RAIL=1`、`MC_IB_TIMEOUT=18`），但源端没有 ionic_4 可选。
- aus 上同样配置跨 rank 无错误，说明 aus 网络允许跨 rail。
- 在 crsuse2 上必须让每个 rank 共享 8 张网卡（RCA 合同的 `ionic_0,…,ionic_7`），由 v3 选目的端
  rail：P rank i 发往 D rank j 时两端都走 `ionic_j`。这是与 B4 的又一处必要差异；客户端校验与
  `preflight_placement.py` 不涉及网卡配置。
- 另：`pgrep -f` / `pkill -f` 的模式会匹配到执行它的 ssh shell 自身，曾导致收尾进程监视错误的
  PID，已改用 `[x]` 写法核对。

## 待办

- 定第二台节点后：把最终镜像 `docker save | docker load` 到该节点，再按 README 运行。
- 运行后：对比 B4 的 `agentx_conc80.json` 与汇总表指标（Total/Output tok/s/GPU、TTFT
  p50/p90、ITL p50/p90、P miss、D 本地复用、P 排队、D 等待 KV）。
