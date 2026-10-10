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

| 06:30 | — | 用户要求：1P(TP8)+1D(TP4) C80，其余与对齐配置一致 | 按 aus 的 B5 设计（`make_capacity_point.py b5`）：D TP4/DP4、GPU 0–3、chunk 16384（每 rank 4096）、max-running 256、graph bs 256；`B4_DECODE_TP=4` 开关，D 容量无 B4 参考只记录；空闲检查只看用到的 GPU |
| 06:32 | 136 | `B4_DECODE_TP=4 B4_GATE_SOFT=1 B4_ANSWER_CHECK=1 setsid scripts/run_unattended.sh` | `runs/b4-crsuse2-136-138-d4-20261009T0632Z/`，日志 `run4-p8d4-136-138.log` |
| 06:39–06:46 | 136/138 | 启动、gate、答案检查、switch、preflight | D TP4/DP4、GPU 0–3、共享 8 网卡；gate 16/16；答案检查 0 传输错误、错 3（同类长文档读错）；D 每 rank KV 2,111,808 |
| 06:46–08:12 | 136/138 | C80 | warmup 18 分钟后正式窗口 3600 秒：完成 7,287、取消 31、错误 0；无传输失败。`MEASUREMENT_COMPLETE` |
| 08:12 | 136 | 收尾失败 | `run_unattended.sh: error reading input file: Stale file handle`：06:33 提交后的 `git rebase` 替换了正在执行的脚本，`stop_after.sh` 未运行；08:15 手动 `run_b4.sh stop`。脚本已改为复制到 `/tmp` 后执行 |
| 08:45 | 登录节点 | 发现：第三轮（P8D8）重跑 measure 时 `sample_engine_metrics.py` / `sample_node_runtime.py` 因输出已存在而退出，`sampling/` 只有 03:33 首次尝试的 3 个样本 | 客户端与诊断日志的分析不受影响；P8D8 缺 D KV 占用的时间序列。`run_b4.py measure` 已改为先把旧采样文件改名、采样器 10 秒内退出即报错 |
| 08:48 | 136 | P8D4 调优：`B4_DECODE_TP=4 B4_SESSION_AFFINITY=prefill B4_DECODE_GRAPH_MAX_BS=64 B4_DECODE_MEM_FRACTION=0.90 B4_TAG=ponly-g64-m090 … run_unattended.sh` | 依据：P8D4 中 D 各 rank KV 占用均值 68/50/83/85%，52% 的样本一热（>90%）一冷（<50%），排队集中在 rank 3；`runs/b4-crsuse2-136-138-d4-ponly-g64-m090-20261009T0848Z/`，日志 `run5-p8d4-tuned-136-138.log` |
| 08:56–11:01 | 136/138 | P8D4 调优运行 | D 每 rank KV 2,378,624（+12.6%）、graph bs 64、mem 0.90、亲和 `prefill`、D radix 开；答案检查 136/136 对；warmup 09:07–09:53（较长），正式窗口 09:53:29–10:53:58：完成 7,990、取消 30、错误 0；`stop_after.sh`（`/tmp` 副本）11:01 正常删除容器 |
| 10-10 04:25 | 登录节点 | 分析 | Total token/s/GPU 21,081（调优前 19,259，P8D8 22,521），Output 172.4（略高于 P8D8）；阻塞 1,597→1,427；D 各 rank 占用 80/73/66/66%。根因：router 对 D 无 token 负载信号（D 无 `kv_block_size`，负载只计派发次数），请求数均匀而 KV 不均。见 `../P8D4-OPTIMIZATION.zh-CN.md` 第 6 节；下一步打开 R2 |
| 10-10 04:33 | 全部 | 节点检查 | 135：GPU 1 被他人 vLLM 进程占 90 GB，其余空闲；136：yihou sanity 占 GPU 0–3；137：limou 的 vLLM TP8 刚启动；138：空闲。用户决定 P 用 138 的 8 卡、D 用 135 的 4 卡 |
| 04:40 | 135/138 | 准备 | 新增 `B4_DECODE_GPUS`（D 用 135 GPU 4–7，NUMA 1；Mooncake 按物理 NUMA 给 GPU 固定本地网卡，GPU 4–7→ionic_4–7）、`B4_R2`、135 的 IP；镜像 138→135（`e13b5843…`，`image-transfer-138-to-135.log`）；kernel/AITER 缓存 136→138（P TP8）、138→135（D TP4） |
| 04:42–04:52 | 138/135 | P8D4 + R2（P 138、D 135 GPU 4–7）`run6-p8d4-r2-138-135.log` | 配置生效（D TP4、HIP 4–7、R2 on、亲和 prefill）。gate 首个请求 HTTP 500：135 的 ionic_7 没有网卡接口、GID index 1 全 0（`Failed to modify QP to RTR … No such device`），D rank 3（GPU 7 固定 ionic_7）不可达。平台问题，需报 IT。已停止编排进程并删除容器，此轮作废 |
| 06:31 | — | 用户与 limou 协商后决定用 137（P）+ 135（D） | 137 limou 容器已停、8 卡空闲、8 张网卡正常；镜像 135→137（`image-transfer-135-to-137.log`，`e13b5843…`）；P TP8 kernel/AITER 缓存 138→137 |
| 06:36–08:21 | 137/135 | P8D4 + R2（`run7-p8d4-r2-137-135.log`） | D 网卡白名单 `ionic_0…6`、HIP 0,4,5,6、R2 on、亲和 prefill；0 传输错误；答案检查 128/128 对（难题重放 4/8 错）；正式窗口 07:14:13–08:14:42：完成 8,881、取消 37、错误 0；08:21 自动收尾 |
| 08:30 | 登录节点 | 分析 | Total token/s/GPU 23,051（P8D8 22,521，+2.4%），Output 191.3；D rank 占用 83.5–84.0%，一热一冷 0.2%；瓶颈转为总容量。见 `../P8D4-OPTIMIZATION.zh-CN.md` 第 7 节 |
| 08:56 | — | 用户选择容量方向：D mem 0.92 + 预留 2048 | 查证后只做 mem 0.92：客户端 `max_tokens` 取自 trace 的输出长度（`weka_trace.py` `Turn.max_tokens`），D 预留 `min(max_tokens, 4096)` 本就约等于真实输出；输出 >2048 的约 10%（p90 1,991），平均每请求只少预留约 100 token（<0.1% 容量），且长输出可能触发 retract |
| 08:20 | 登录节点 | 分析 | 见 `../RESULT-P8D4.zh-CN.md`：D KV 容量不足（1,597 个请求等分配，D 复用 89%→35%），完成 −31.5%，每 GPU Output −8.8%、Total −14.5% |

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
