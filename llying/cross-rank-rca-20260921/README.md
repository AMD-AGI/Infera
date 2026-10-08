# Cross-Rank Mooncake RCA 实验包

本目录只用于 Mooncake cross-rank 根因复现、单变量矩阵、根因修复和最终验收。
Router rank affinity、应用层重试、CPU/GPU staging 均不作为修复。

## 固定配置

- 代码：`baseline/Infera`，初始提交
  `83e0f6c86cce34718f369d8669b750fa812c62d0`
- Prefill：`crsuse2-m2m-137` / `10.245.153.247`（2026-09-28 起；
  2026-09-21 的复现在 `crsuse2-m2m-138` / `10.245.157.237`）
- Decode：`crsuse2-m2m-136` / `10.245.154.168`
- 基底镜像：`infera-sglang:v0519-yihou-0917-nextnfix-hicache`
- 基底 image ID：
  `sha256:fd7220a57b7d3b58efd875c41f7a9ef46b93469581102d96cbeb6f5451e91d35`
- 修复镜像（v3，最终版）：`infera-sglang:v0519-yihou-0917-nextnfix-hicache-mcdestpin-ibto18`，
  `sha256:88286215b79d43306dce88dcdf299a13d939c6b708691dc192ca7c7c33c518a5`
  （`fix/`，只替换 Mooncake `engine.so` 并设置 `MC_ENABLE_DEST_LOCAL_RAIL=1`、
  `MC_IB_TIMEOUT=18`）
- P8D8、C80、warmup/lane=1、profiling=1200s
- HiCache off、`PD_DP_RANK_AFFINITY=0`
- `RDMA_DEVICE=MC_TE_FILTERS=ionic_0,...,ionic_7`
- `MC_ENABLE_DEST_DEVICE_AFFINITY=1`、`MC_GID_INDEX=1`

`config/config.rca.p8d8.sh` 在加载 packup P8D8 配置前固定所有差异，并在加载后
检查有效值；`config/config.rca-fix.p8d8.sh` 只把镜像换成修复镜像。
节点对由 `RCA_PREFILL_NODE` / `RCA_DECODE_NODE` 选择，默认 137→136（topology
`config/topology.rca.tsv`），其它节点对用 `config/topology.rca-<P>-<D>.tsv`
（已有 138→136、136→138）；配置会导出所选的 `RCA_TOPOLOGY`。Python 工具通过
`scripts/rca_nodes.py` 读取同一节点对和镜像（`RCA_PREFILL_NODE`、
`RCA_DECODE_NODE`、`RCA_IMAGE`、`RCA_IMAGE_ID`，配置文件会导出它们）。
`run_reproduction.sh` 的 HCA 计数器采样间隔由 `RCA_COUNTER_INTERVAL` 设定，默认 10 秒。
`config/config.rca-fix-ibto24.p8d8.sh` 是诊断用配置（两端 `MC_IB_TIMEOUT=24`），不是修复。

## 目录

- `config/`：固定配置和 topology
- `scripts/`：provenance、HCA counter、live assertion、运行与分析工具
- `fix/`：Mooncake 补丁、修复镜像 Dockerfile 和构建脚本
- `provenance/`：清场前和运行时的软件/硬件证据
- `operations/`：镜像迁移、清场、恢复和进行中任务记录
- `runs/<RUN_ID>/`：每次 fresh AgentX 运行的完整产物
- `matrix/`：GPU pair 的逐字节验证、并发压力和 host memory 机理结果
- `REPORT.zh-CN.md`：根因结论和验收报告

每次运行的原始产物不进 git，只保留在共享文件系统上的本目录里，包括
`runs/*/bench/aiperf_artifacts/`、`runs/*/launch/server-logs/`、`runs/*/logs/`、
`runs/*/counters/timeseries.jsonl` 和 `cache/`。报告引用的是 `runs/*/analysis/`
中的汇总。本目录原在 `bench/glm5p2_pd/results/cross-rank-rca-20260921`（该路径
被仓库忽略），现在那里是指向本目录的软链接，历史记录中的旧路径仍然有效。

## 运行顺序

长任务都在 Prefill 节点上用 `setsid nohup` 运行，日志写在本目录。节点上可能有其它
用户的容器：`scripts/clear_nodes.sh` 会停止所有占用 GPU 的容器，未经明确授权
不得运行；开跑前只用 `docker ps` 和 `rocm-smi` 确认节点空闲。

```bash
ROOT=/home/liyingli/bench_agentx/baseline/Infera/llying/cross-rank-rca-20260921

# 在 137 上构建修复镜像并复制到 136
bash "$ROOT/fix/build_image.sh" crsuse2-m2m-136

# 基底镜像 C80 基线 / 修复镜像 C80
bash "$ROOT/scripts/run_reproduction.sh"
CONFIG="$ROOT/config/config.rca-fix.p8d8.sh" bash "$ROOT/scripts/run_reproduction.sh"

# 结束后只删除本实验容器
source "$ROOT/config/config.rca.p8d8.sh"
bash "$BENCH_DIR/stop.sh" "RCA_PREFILL_NODE=$PREFILL_NODE" "RCA_DECODE_NODE=$DECODE_NODE" \
  "CONFIG=$ROOT/config/config.rca.p8d8.sh" "TOPOLOGY=$RCA_TOPOLOGY"

# 无人值守连续运行（在 Prefill 节点上），每轮前检查空闲与镜像、跑完 stop 并等显存释放
RCA_PREFILL_NODE=crsuse2-m2m-136 RCA_DECODE_NODE=crsuse2-m2m-138 RCA_COUNTER_INTERVAL=1 \
  setsid nohup bash "$ROOT/scripts/run_fix_chain.sh" config.rca-fix.p8d8.sh:fix-06 \
  > "$ROOT/operations/<name>-chain.log" 2>&1 < /dev/null &
```

新镜像第一次启动前，把两节点的 `/tmp/aiter-jit-$(id -u)/<基底 image ID>`
复制到 `<新 image ID>` 目录，否则 AITER JIT 冷编译会拖长启动（`run_fix_chain.sh`
会自动做）。stop.sh 在 136 上返回 1 是平台 `gpuagent` 持有 KFD 上下文（0 显存），无害。

必须先检查 `live-config.json`。任何 run 出现 CQE12、retry exhausted、ACK timeout、
byte mismatch 或 worker fatal error，都先保存现场，不得直接重跑覆盖。

## 离线分析

```bash
RUN="$ROOT/runs/<RUN_ID>"
python3 "$ROOT/scripts/analyze_reproduction.py" "$RUN"
```

分析产物写入 `$RUN/analysis/`。

## Pair matrix

`run_pair_matrix.py` 串行运行，逐 pair 采 HCA counter；
`run_concurrent_matrix.py` 每节点一个共享容器、每 wave 8 pair 并发，counter
按 wave 归因，`--worker-env KEY=VALUE` 给 worker 追加环境变量，`--fan-in K`
让每 K 个 source 同时写同一个 destination GPU。worker 与 SGLang rank 一样能看到
全部 8 张 GPU，buffer 的 Mooncake location 就是物理 GPU 号。两者都用
`--dry-run` 打印计划，硬件执行必须显式加 `--confirm-exclusive`。

```bash
cd "$ROOT/scripts"
export RCA_IMAGE=infera-sglang:v0519-yihou-0917-nextnfix-hicache-mcdestpin-ibto18
export RCA_IMAGE_ID=sha256:88286215b79d43306dce88dcdf299a13d939c6b708691dc192ca7c7c33c518a5
python3 run_pair_matrix.py --output "$ROOT/matrix/<name>" \
  --profile batch --policies auto --confirm-exclusive
python3 run_concurrent_matrix.py --output "$ROOT/matrix/<name>" \
  --profile soak --seconds 60 --policies auto --confirm-exclusive
```

最终结果门禁：

```bash
python3 "$ROOT/scripts/certify_results.py" \
  --matrix "$ROOT/matrix/<fixed-auto-matrix>" \
  --agentx-run "$ROOT/runs/<fixed-agentx-run>"
```
