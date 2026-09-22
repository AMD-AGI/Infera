# GLM-5.2 MXFP4：2P1D AgentX conc sweep

参考 [`yihou/glm52.p8d8.agentx-sweep.packup_20260920`](../../yihou/glm52.p8d8.agentx-sweep.packup_20260920/README.md)
的最终稳定配置，将 1P1D 扩展为 **2 个 Prefill + 1 个 Decode**，每个实例
TP8/DP8/DPA，共 **24 张 MI355X**。默认扫描 `80 112 144 192 256`，
每档预热 `10 requests/lane`，正式测量 `3600 s`，整轮共用同一次部署。

本轮使用 137/138 做 Prefill、136 做 Decode，运行编号 `2p1d-20260921T151735Z`，
五档均已于 **2026-09-22 完成**。正式发送窗口各为 3600 秒，整轮共用同一部署。
本轮总吞吐峰值在 **C144：18,356.23 token/s/GPU**，输出吞吐为 152.81 token/s/GPU；
继续增至 C256 后，总吞吐降至 14,346.41，TTFT p50 从 8.98 秒升至 68.22 秒。
最终结果按参考套件布局写入 [`results/`](results/README.md)，包含 [CSV](results/results.csv)、
[完整指标表](results/sweep_results.md) 和 [曲线图](results/sweep.png)。
配置并发与 Decode 实际运行并发的差距、1P1D 对比及 KV 队列证据，见
[Decode 并发分析报告](DECODE_CONCURRENCY_REPORT.md)。
本轮容器已于 2026-09-22 00:21 UTC 清理；Prefill HiCache 显存会异步释放，复用节点前需重新检查。
完整原始数据、cache、运行日志和开发测试留在 [`.tmp/`](.tmp/README.md)，由 Git 忽略。

PD 分离修复的来源、作用、同步内容和验证边界见
[PD_FIXES_REPORT.md](PD_FIXES_REPORT.md)。参考镜像的额外补丁保存在 [`patches/`](patches/README.md)。

## 默认部署与对比口径

| 实例 | 节点 | data IP | HTTP 端口 | GPU |
|---|---|---|---|---|
| prefill-0 | crsuse2-m2m-137 | 10.245.153.247 | 29001 | 0–7 |
| prefill-1 | crsuse2-m2m-138 | 10.245.157.237 | 29002 | 0–7 |
| decode-0 | crsuse2-m2m-136 | 10.245.154.168 | 29003 | 0–7 |
| router / etcd | crsuse2-m2m-137 | 10.245.153.247 | 28000 / 22379 | — |

节点选自用户指定的 135/136/137/138；本轮排除 GPU 被占用且 ionic_7 GID 为零的 135。运行前通过 `scripts/check.sh` 检查资源。
更换节点时修改本目录的 [`topology.tsv`](topology.tsv)，并让 `CONTROL_NODE`
与其中一行的 node 字段一致。worker 端口按 TSV 行号分配，不要写死 Decode 为 29002。

[`config.sh`](config.sh) 是部署和压测共同使用的配置来源：

| 参数 | 默认值 / 说明 |
|---|---|
| SSH 账号 | `SSH_USER=xiaobche`，通过 `SSH_OPTS` 的 `-l` 传递 |
| 镜像 | `infera-sglang:v0519-yihou-0917-nextnfix-hicache` |
| 镜像 ID | `sha256:fd7220a57b7d3b58efd875c41f7a9ef46b93469581102d96cbeb6f5451e91d35`；启动前逐节点核对 |
| 模型 | `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` |
| P/D shape | 每实例 TP8、DP8、EP1、DPA 开启 |
| P/D max-running / graph max BS | 均为 256，覆盖最高并发档位 |
| Prefill HiCache / Decode HiCache | 开 / 关 |
| Decode MTP | EAGLE，5 steps / 6 draft tokens / topk 1 |
| 模拟接受长度 | `DECODE_SIMULATE_ACC_LEN=3.61` |
| IndexShare | `index_share_for_mtp_iteration=false`，沿用参考的稳定性设置 |
| DSA | Prefill/Decode 均为 `tilelang`，top-k 使用镜像默认；保留 fused indexer |
| KV transfer | Mooncake，router 同 DP rank 路由；GPU i → ionic_i，GID index 1 |
| InferenceX | 固定 commit `918524ff94045b3f091115f1051c22a8588edf2b` |

镜像包含参考使用的 NextN fusion 和 HiCache 补丁，本套件不重新构建或在运行时打补丁。
镜像 ID 来自参考的 `logs/launch-summary.txt`。若有意换镜像，应同时设置其
`EXPECTED_IMAGE_ID`；三台节点的实际 ID 仍必须一致。

模拟 acceptance 会强制接受长度，可能产生乱码，因此这些数据用于性能比较，
不表示生成正确性。需要真实 acceptance 时，使用新的运行传入
`DECODE_SIMULATE_ACC_LEN=`，并单独标注结果口径。正式报告同时列出包含输入和输出
token 的总吞吐，以及仅输出吞吐；两者都除以 **24**。输入吞吐包括命中缓存的
输入 token，不能解释成实际计算了同等数量的 Prefill token。

## 运行方法

需要：三台空闲的 8-GPU 节点、相同的指定镜像、模型和共享目录、Docker 权限，
以及执行机 → 控制节点、控制节点 → 所有节点（含自身）的免密 SSH。
当前默认使用 `xiaobche`，执行机账号 `cyao1002` 和远端账号均需能写共享 `.tmp/` 下的输出和缓存目录。
首次运行 InferenceX 会拉取固定版本和依赖，控制节点需要访问相应 Git/HF 资源。

```bash
cd /shared_nfs/yaoc/work/GLM-5.2-pd-opt/Infera-isl-debug/yaocheng/2p1d-sweep

# 本地解析配置和拓扑，无 SSH、无服务启动。
bash scripts/run_full_sweep.sh DRY_RUN=1

# 环境检查：SSH、Docker、GPU 占用、镜像 ID、路径、端口、RDMA rails，
# 并用不挂载 GPU 的只读 CPU 容器检查镜像中的 PD 补丁。
bash scripts/check.sh

# 可独立检查镜像补丁；不要求 GPU 空闲，不启动推理服务。
bash scripts/verify_pd_fixes.sh

# 一键完整运行，可从登录节点执行；编排和 AgentX 客户端在 CONTROL_NODE 上运行。
bash scripts/run_full_sweep.sh
```

所有入口接受 `KEY=VALUE` 参数。有空格的值须作为一个参数传入，例如：

```bash
# 默认已使用 xiaobche；更换时设置 SSH_USER。
# 若另行覆盖 SSH_OPTS，也要保证其中 -l 的用户与 SSH_USER 一致。
bash scripts/check.sh SSH_USER=USERNAME

# 增加低并发档位，保持 3600 秒和 10/lane。
bash scripts/run_full_sweep.sh 'SWEEP_POINTS=32 40 56 80 112 144 192 256'

# 短跑仅验证运行链路；不与正式长测混用。
bash scripts/run_full_sweep.sh 'SWEEP_POINTS=8' DURATION=120 \
  AGENTX_WARMUP_REQUESTS_PER_LANE=1
```

完整五档仅 profiling 就需要 **5 小时**，还需加上加载模型、RDMA preflight、预热、
排空和结果聚合。本轮 C80/C112/C144/C192 单档实测墙钟分别约 **88/86/91/99 分钟**，
其中预热约 **16/15/19/26 分钟**。C256 有效运行约 **2 小时 12 分钟**，其中预热约 56 分钟，
经过两次人工恢复；此前另有约 39 分钟的 C256 预热被中止。
本轮从 2026-09-21 15:17 到 2026-09-22 00:19 完成发布，约 **9 小时 2 分钟**。
高并发不能按低并发线性估时；参考 1P1D 的 C256 仅预热就用了约 3 小时 10 分钟，
参考整轮约 13.5 小时。
长任务可用以下方式启动和观察：

```bash
RUN_ID="2p1d-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p .tmp/logs
nohup bash scripts/run_full_sweep.sh "RUN_ID=$RUN_ID" \
  > ".tmp/logs/$RUN_ID.console.log" 2>&1 < /dev/null &
tail -f ".tmp/logs/$RUN_ID.console.log"
# 进度：.tmp/results/$RUN_ID/status.txt
# 运行中每完成一档，就更新 .tmp/results/$RUN_ID/summary.md 和 results.csv。
```

启动前会冻结配置到 `.tmp/results/<RUN_ID>/config.resolved.sh`，后续 launch、engine、
AgentX 和 stop 都读取它；修改原 `config.sh` 不会改变已开始的这轮配置。
所有节点必须能访问 `.tmp/` 的相同绝对路径。默认路径统一如下：

| 配置 / 内容 | 默认路径（相对本目录） |
|---|---|
| `SWEEP_TMP_DIR` | `.tmp/` |
| `RESULTS_DIR`：最终聚合 JSON、CSV、rail 记录和报告 | `results/` |
| `RUN_DIR`：配置快照、日志、preflight、按档指标和汇总 | `.tmp/results/<RUN_ID>/` |
| `RAW_DIR`：完整 AgentX/AIPerf 原始结果 | `.tmp/raw/<RUN_ID>/` |
| `AGENTX_CACHE_DIR`：AIPerf、HF、UV、pip、XDG 缓存 | `.tmp/cache/agentx/` |
| `INFERENCEX_DIR`：固定版本 checkout 及获取过程临时目录 | `.tmp/cache/InferenceX/` |
| `AITER_JIT_CACHE_ROOT`：按节点、镜像隔离的编译缓存 | `.tmp/cache/aiter/<node>/<image-id>/` |
| 部署锁 | `.tmp/.deployment.lock` |

Python 脚本默认关闭字节码写入，避免在 `scripts/` 下生成 `__pycache__/`。
AgentX 容器的工作目录位于对应 `.tmp/raw/<RUN_ID>/cNNN/` 下；其 `tmp/` 在容器内
挂载为短路径 `/ax-tmp`，避免 AIPerf 的 Unix socket 超过 107 字节路径限制，实际文件仍在 `.tmp/`。
默认原始结果和 cache 都写入本共享目录的 `.tmp/`，需为这部分数据预留空间。

## 脚本分工

| 脚本 | 功能 |
|---|---|
| `scripts/run_full_sweep.sh` | 冻结配置 → SSH 到控制节点 → 节点检查 → 两条 P→D 路径的 Mooncake WRITE preflight → 部署 → sweep → 停止服务 |
| `scripts/check.sh` | 检查三节点；占用、旧容器、镜像/rail 不匹配、路径、端口或 PD 补丁缺失均在启动前报错 |
| `scripts/verify_pd_fixes.sh` | 用只读 CPU 容器检查 router rank-affinity CLI 与 SGLang PD/DSA/HiCache/NextN 修复；JSON 输出到 `.tmp/` |
| `scripts/publish_results.py` | 将已完成且通过验证的档位导出到 `results/`，保留参考的 cNNN/JSON/CSV/rails 布局 |
| `scripts/sweep_driver.sh` | 对已部署服务逐档压测；每档前后检查健康、容器身份和错误计数，归档小型结果并汇总 |
| `scripts/audit.py` | 从三台 worker 统计 RDMA retry/WQE、GPU memory fault、Fatal Python、OOM；检查容器是否重启和 router rank affinity |
| `scripts/summarize.py` | 校验 2P1D/24-GPU 元数据、并发、测量时长和每卡吞吐，再生成 CSV/Markdown |
| `scripts/bench-harness/launch.sh` | 启动 etcd、两个 Prefill、一个 Decode、router，并等待发现和健康检查 |
| `scripts/bench-harness/agentx_bench.sh` | 单档 AgentX；对照实际容器参数生成 `runtime.env`，执行 InferenceX/AIPerf |
| `scripts/bench-harness/stop.sh` | 停止指定配置前缀的服务和客户端，等待 worker graceful stop，打印剩余 GPU 状态 |

`RUN_PREFLIGHT=1`、`STOP_AFTER_SWEEP=1` 为默认值。
`POINT_TIMEOUT=21600` 限制单档整个客户端过程最多 6 小时（包含预热/测量/排空）。
任一档 benchmark 失败、服务重启、健康检查失败、新增 rail/GPU/OOM 错误，都会停止后续档位。
默认退出时清理本轮服务；失败证据留在输出目录。

需要手动保留部署继续测试时，在首次启动指定 `STOP_AFTER_SWEEP=0`。
继续测量只能选择尚未开始的档位，并且必须是原容器、原启动时间、原镜像；
driver 应在控制节点运行：

```bash
# 将 <RUN_ID> 替换为实际目录，以下命令在控制节点执行。
RUN_DIR="$PWD/.tmp/results/<RUN_ID>"
bash scripts/sweep_driver.sh "CONFIG=$RUN_DIR/config.resolved.sh" \
  "TOPOLOGY=$RUN_DIR/topology.tsv" 'SWEEP_POINTS=192 256'

# 手动停止也使用该轮保存的配置。
bash scripts/bench-harness/stop.sh "CONFIG=$RUN_DIR/config.resolved.sh" \
  "TOPOLOGY=$RUN_DIR/topology.tsv"
```

driver 不会因为目录存在就跳过该档；已有目录会报错。失败档建议新开一轮，
保留原始证据。停止后 GPU 显存释放可能仍需较长时间，下一轮启动会重新检查占用。

## 当前集群的可写目录

计算节点将 `/shared_nfs` 挂载为只读。本轮为此套件目录创建了独立 NFS 可写挂载：

```text
/mnt/m2m_nobackup/xiaobche/2p1d-sweep/shared-kit
```

它指向同一个 NFS 套件目录；通过该别名写入的 `.tmp/` 和 `results/`，可以直接
从本 checkout 的相应目录读取。模型仍使用原有只读路径。该挂载已在 136/137/138
准备好，运行目录为两个执行账号配置了访问 ACL。

从控制节点 137 启动时，本轮使用以下覆盖：

```bash
RW_KIT=/mnt/m2m_nobackup/xiaobche/2p1d-sweep/shared-kit
bash scripts/run_full_sweep.sh \
  "SWEEP_TMP_DIR=$RW_KIT/.tmp" "RESULTS_DIR=$RW_KIT/results"
```

## 最终结果

```text
results/
  run.json                     运行编号、拓扑、镜像和测量口径
  results.csv                  按参考表头导出，P/D GPU 数为 16/8
  sweep_results.md             吞吐、TTFT、ITL、完成数及错误数
  c080/, c112/, c144/, c192/, c256/
    agentx_conc<N>.json
    profile_export_aiperf.csv   原始文件可用时复制
    rails-c<NNN>.txt            三个 worker 的前后错误计数
```

只有完成测量且通过校验的档位会进入最终结果；失败和未完成记录留在 `.tmp/`。
本轮 C256 的初始预热发生 HTTP/KV 等待异常，经过人工释放失败等待后继续完整压力预热；
详细恢复条件、错误计数口径和服务端指标限制见 [`measurement_notes.md`](results/measurement_notes.md)。
已发布结果绑定一个 RUN_ID，避免不同部署的数据混到同一份结果中。

## 输出目录

```text
.tmp/results/<RUN_ID>/
  config.resolved.sh, topology.tsv, source-sha256.txt
  status.txt, orchestrator.log, check.log, launch.log, stop.log
  REMOTE_ARTIFACTS.txt            完整原始数据路径与执行节点
  preflight-prefill-{0,1}/       两条 P→D 路径的报告
  patch-check/                  各节点镜像补丁检查结果
  launch/                       worker/router 健康、server-info、服务日志
  deployment.json               用于保证整轮共用同一部署
  c080/, c112/, .../
    agentx_conc<N>.json          AgentX 聚合指标
    status.json, metrics.json, exit-codes.txt
    before.json, after.json      服务身份和累计错误计数
    runtime.env, service/        实际运行配置与硬件快照
    console.log, runner.log, benchmark.log, benchmark_command.txt
    aiperf-summary/              可用时保存 profile_export_aiperf.csv
    REMOTE_ARTIFACTS.txt         该档的完整原始数据位置
  results.csv, summary.md        只汇总已完成且验证通过的档位
```

逐请求 JSONL 等大文件保存在 `.tmp/raw/<RUN_ID>/`，服务日志保存在
`.tmp/results/<RUN_ID>/launch/server-logs/`，每档的小型指标和审计文件复制到
`.tmp/results/<RUN_ID>/cNNN/`。归档所需结果后再清理对应的运行目录。

```bash
# 对已有结果重新汇总；不连接服务，不发请求。
python3 scripts/summarize.py .tmp/results/<RUN_ID>
```

中途添加的离线测试位于 `.tmp/tests/`，历史验证记录和运行方法见
[`.tmp/README.md`](.tmp/README.md)。
