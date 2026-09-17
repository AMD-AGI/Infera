# GLM-5.2 multi-P/D baseline

这是一套人工分步执行的 GLM-5.2 Prefill/Decode baseline。它按 topology
启动任意数量的 Prefill 和 Decode worker，用 etcd 做发现、Infera router
做入口，并保留每一阶段的检查与产物。脚本不会占用集群锁；同一批节点只应有
一个操作者。

## 前置条件

- 所有节点为同构 ROCm GPU 节点，GPU 数量与 `NODE_GPU_COUNT` 一致。
- 控制机到 topology 中所有节点可免交互 SSH；远端可运行 Docker、
  `rocm-smi`、RDMA 设备及 `/dev/kfd`、`/dev/dri`、`/dev/infiniband`。
- Infera 仓库、此 bench、模型、运行结果和缓存路径在相关节点上以相同绝对
  路径可见。推荐共享文件系统。
- `MODEL` 指向每台 worker 与 control node 均可读的 GLM-5.2 权重目录。
- data-plane IP 对应 RDMA NIC；`RDMA_DEVICE`、`MC_GID_INDEX` 和
  `MC_TE_FILTERS` 与集群配置一致。
- build node 能读取目标 Infera 仓库作为 Docker build context，并能访问
  官方 base image 和源码依赖。
- 运行 `analyze_agentx.sh` 的机器需要安装 Python `matplotlib`。

## 一次性配置

```bash
cd bench/glm5p2_pd
```

本目录只保留一个自包含的 `config.sh` 和一个 `topology.tsv`，不再通过
example、local override、profile 多层 source 合并配置。直接编辑
`config.sh` 中的 `MODEL`、`CONTROL_NODE`、镜像、GPU、端口与 RDMA 设置。
最小 topology 是一台 Prefill 和一台 Decode：

```text
role	node	data_ip
prefill	worker-01	10.245.0.11
decode	worker-02	10.245.0.12
```

每个 node 和 data IP 只能出现一次；行号决定 worker 实例名和端口偏移。
`CONTROL_NODE` 必须出现在 topology 中。当前 `config.sh` 直接包含
P8+DPA / D8+DPA 配置。`tools/topology.py` 只校验这三列并提供
`rows`、`nodes`、`count`、`node-ip` 查询；GPU、端口和容器参数由实际使用
它们的部署脚本计算。

除 `build_image.sh` 可选的首个 `build|distribute|verify|all` 子命令外，
部署和评测脚本仍可用命令行 `KEY=VALUE` 临时覆盖 `config.sh` 中的单项值；
允许空值的选项可用 `KEY=` 关闭。

## 执行顺序

### 1. 检查节点

```bash
./check_nodes.sh crsuse2-m2m-136 crsuse2-m2m-140
# 也可以传逗号分隔列表
./check_nodes.sh crsuse2-m2m-136,crsuse2-m2m-140
```

`check_nodes.sh` 是完全独立的只读检查，不读取 `config.sh`、`topology.tsv`
或其他 helper。它逐节点检查 SSH、Docker daemon、每张 GPU 的 VRAM% 与
GPU 利用率。任一能力不可用或 GPU 不空闲时返回非零；发现名称中包含
prefill、decode、router 或 etcd 的已有容器时只打印 WARNING，由操作者确认。
空闲阈值可通过 `GPU_IDLE_VRAM_PCT` 和 `GPU_IDLE_UTIL_PCT` 调整。

### 2. 构建并分发镜像

```bash
./build_image.sh
```

这是单阶段构建：直接以目标 Infera 仓库为 context 构建
`deploy/docker/Dockerfile.sglang`。默认 base 为
`lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260916`。不构建额外 fork
base，也不依赖其他 benchmark 仓库。

常用覆盖：

```bash
./build_image.sh build \
  IMAGE=infera/engine-sglang:test \
  BUILDER_NODE=worker-01
./build_image.sh distribute IMAGE=infera/engine-sglang:test
./build_image.sh verify IMAGE=infera/engine-sglang:test
```

默认 base 已在 `config.sh` 中设置；Dockerfile 默认启用
`BUILD_MOONCAKE=1` 和 `APPLY_SGLANG_DSA_PATCHES=1`，不需要重复传入。
`BUILD_ARGS` 中每一项必须为 Docker `ARG=value`。脚本验证每台 topology
节点上的镜像 ID 一致，并检查 Infera/SGLang import 和 router 二进制。

### 3. 多节点 preflight

```bash
IMAGE=infera-sglang:v0519-baseline \
  ./preflight.sh crsuse2-m2m-136 crsuse2-m2m-140
```

`preflight.sh` 不读取 benchmark 配置/topology，也不调用 `check_nodes.sh`。
它只负责在传入节点上并发启动同一个 `infera.tools.preflight` 多节点任务；
核心 preflight 自己扫描节点对并生成原始结果。默认运行 network 和 Mooncake
WRITE byte-verification；可用 `PREFLIGHT_MOONCAKE_OPCODE` 显式覆盖。产物位于
`results/<UTC>-preflight/`：每个 rank 的日志、每节点 JSON、Mooncake 中间
数据以及 rank 0 生成的 `infera_preflight_report.html`。不再有自定义 pair
汇总或 `validation.json` gate。

### 4. 启动服务

```bash
./launch.sh OUT_DIR=results/launch
```

顺序为 etcd → 全部 P/D worker → worker health → router →
router health/discovery。节点检查需要由操作者在启动前显式运行；
`engine.sh` 由 launch 经 SSH 远程调用，并按 role
分别读取 `PREFILL_*`、`DECODE_*` 参数。检查快照和失败日志写入 launch
输出目录；每个 worker 的完整 Docker 日志持续写入
`OUT_DIR/server-logs/<instance>.log`，容器删除后仍保留。

每个 worker 节点还会把 AITER JIT 产物保存在
`/tmp/aiter-jit-<uid>/<image-id>` 并挂载为容器内 `/aiter-jit`。cache 按实际
Image ID 隔离，重启相同镜像时复用，切换镜像时不会读取旧 kernel；可用
`AITER_JIT_CACHE_ROOT` 修改根目录。成功后服务保持运行。

### 5. correctness

模拟 MTP acceptance 不用于 correctness。先按空值覆盖重启：

```bash
./stop.sh
./launch.sh DECODE_SIMULATE_ACC_LEN= OUT_DIR=results/launch-correctness
./eval/smoke.sh DECODE_SIMULATE_ACC_LEN= OUT_DIR=results/smoke
./eval/gsm8k.sh DECODE_SIMULATE_ACC_LEN= OUT_DIR=results/gsm8k
./eval/long_context.sh DECODE_SIMULATE_ACC_LEN= \
  TOKENS=250000 OUT_DIR=results/long-context
```

smoke 检查 discovery、普通 chat、tool call 和并发 burst；GSM8K 使用固定
InferenceX checkout 的官方 task/score validator；long-context 保存完整响应
和 token/timing 摘要。

### 6. AgentX

如需与旧 performance 数据保持同一默认模拟设置，先恢复默认服务：

```bash
./stop.sh
./launch.sh OUT_DIR=results/launch-performance

./agentx_bench.sh CONC=64 DURATION=1200 OUT_DIR=results/sweep/c64
```

该对齐点使用 P8+DPA/D8+DPA、decode MTP、P/D graph/max-running 64、
HiCache 关闭，以及每 lane 1 个 warmup request；这些值也是示例配置的默认值。

每个 point 保存 aggregate JSON、`runtime.env`、runner log，以及 router、
worker、container 和硬件快照。适配器从 live service 获取 P/D worker 数、
TP/EP/DP/DPA、HiCache、MTP、模型、镜像 ID、GPU 型号和 DRAM，发现配置
不一致时停止。AIPerf 的 venv、uv 和 Hugging Face cache 默认放在 control
node 的 `/tmp/inferencex-agentic-$USER`，避免把可再生成的依赖写入共享
home；可通过 `AGENTX_CACHE_DIR` 覆盖。

### 7. 收集、分析和 plot

```bash
./analyze_agentx.sh RESULT_DIR=results/sweep
```

生成 `results.csv` 和 matplotlib 绘制的 `pareto.png`。每 chip throughput
按 Prefill GPU 与 Decode GPU 总数计算；不同 topology 分曲线绘制。图中同时
加载 `tools/ref/InferenceX_GLM-5.2_interactivity.csv` 的官方 InferenceX
参考数据，并按 hardware、framework、precision 和完整 P/D shape 分组。

正常绘图只读取已保存的 reference，不访问网络。需要更新时运行：

```bash
python3 tools/update_inferencex_ref.py
```

该命令从 InferenceX 公共 API 重建 reference CSV，并在文件头记录抓取日期、
最新 benchmark 日期和记录数。

### 8. 停止

```bash
./stop.sh
```

worker 的 `docker stop` 默认等待 300 秒，让 HiCache host pool 完成注销；
超时或 graceful stop 失败后才强制删除。最后打印各节点剩余 GPU 进程和内存。

## InferenceX 获取

`agentx_bench.sh` 和 `eval/gsm8k.sh` 会调用
`tools/ensure_inferencex.py`。默认目录是本套件的 `.cache/InferenceX`，
默认仓库是官方 `https://github.com/SemiAnalysisAI/InferenceX.git`，固定
commit 为 `918524ff94045b3f091115f1051c22a8588edf2b`。可用
`INFERENCEX_DIR`、`INFERENCEX_REPOSITORY`、`INFERENCEX_REF` 覆盖，但 ref
必须是完整 40 位 commit。

首次获取在同目录临时 checkout 完成后原子 rename；锁文件串行化并发调用。
已有 checkout 会校验 origin、拒绝 dirty tree、在对象缺失时 fetch，然后
detach checkout 到固定 ref，并初始化仓库固定的递归 submodule。InferenceX
checkout 本身不是本仓库的 submodule。

## Inherited mitigations

以下默认值原样继承旧 workflow，只为结果可比；它们不是已确认的根因修复，
在 v0.5.19 上的性能影响未知，需后续逐项 A/B：

- `SGLANG_ENABLE_FAILED_SESSION_PROBE=1`
- `MC_DISABLE_HIP_TRANSPORT=1`
- `MOONCAKE_DISABLE_HIP_DMABUF=0`
- `SGLANG_OPT_USE_TOPK_V2=false`
- `NCCL_IB_DISABLE=1`
- `MC_ENABLE_DEST_DEVICE_AFFINITY=1`

同时保留原有 `RDMAV_FORK_SAFE=1`、AITER/fused kernel、KV-aware router、
Mooncake RDMA、Decode MTP 与模拟 acceptance 默认设置；本对齐配置的 P/D
HiCache 均关闭。

本 baseline 不包含离线容量投影/仿真、历史报告或运行数据；不包含按 P/D role
设置 ROCr scratch 与 PyTorch allocator 的显存 workaround；不包含 router
按 P/D attention rank 固定路由的功能；也不在 bench 中控制 fork 专用的
SGLang rejection build 开关。相关 v0.5.19 修复由 Dockerfile patch set 管理。

## 高价值故障排查

- SSH/Docker/GPU：对目标节点运行 `./check_nodes.sh NODE...`；根据逐项
  PASS/ERROR 和已有服务容器 WARNING 处理。
- RDMA/Mooncake：查看 preflight 输出目录中的
  `infera_preflight_report.html`、每节点 JSON 和 `rank-*.log`。
- worker/router：查看 launch 的 `failures/`、health JSON、`workers.json`，
  再执行 `ssh NODE docker logs CONTAINER`。
- 结果缺失：AgentX 先查 `runner.log` 与 `runtime.env`，确认 control node
  可读模型、InferenceX、输出目录和 cache 的相同绝对路径。
- 停止：正常情况等待最多约 300 秒。强制路径仍失败时保留
  `docker inspect`/`rocm-smi` 证据后人工处理，不要立即复用仍在 drain 的 GPU。

## 可复现：2P1D AgentX C8 workflow

下面是本次实际使用的节点、镜像、参数和执行顺序。命令从当前 checkout
执行；AIPerf 原始文件和聚合过程放在 control node 的本地 NVMe，避免共享
`/home` 空间不足。共享目录只保存 launch/preflight 记录、控制台日志和最终
小型结果。

```bash
set -euo pipefail

BENCH_DIR=/home/liyingli/bench_agentx/baseline/Infera/bench/glm5p2_pd
CONTROL_NODE=crsuse2-m2m-140
RUN_ID="2p1d-c8-warmup1-nohicache-$(date -u +%Y%m%dT%H%M%SZ)"
SHARED_OUT="$BENCH_DIR/results/$RUN_ID"
REMOTE_OUT="/mnt/m2m_nobackup/$USER/agentx-results/$RUN_ID"
REMOTE_CACHE="/mnt/m2m_nobackup/$USER/agentx-cache"

cd "$BENCH_DIR"
mkdir -p "$SHARED_OUT"

# topology.2p1d.tsv:
# prefill crsuse2-m2m-136 10.245.154.168
# prefill crsuse2-m2m-137 10.245.153.247
# decode  crsuse2-m2m-140 10.245.159.30
./check_nodes.sh \
  crsuse2-m2m-136 crsuse2-m2m-137 crsuse2-m2m-140

# preflight 必须在 workers 启动前运行。只检查实际的两条 P -> D 路径；
# preflight.sh 默认执行 Mooncake WRITE byte-verification。
IMAGE=infera-sglang:v0519-baseline \
HOST_RDMA_LIB=/lib/x86_64-linux-gnu/libionic.so \
OUT_DIR="$SHARED_OUT/preflight-136-140" \
  ./preflight.sh crsuse2-m2m-136 crsuse2-m2m-140

IMAGE=infera-sglang:v0519-baseline \
HOST_RDMA_LIB=/lib/x86_64-linux-gnu/libionic.so \
OUT_DIR="$SHARED_OUT/preflight-137-140" \
  ./preflight.sh crsuse2-m2m-137 crsuse2-m2m-140

./launch.sh \
  TOPOLOGY=topology.2p1d.tsv \
  OUT_DIR="$SHARED_OUT/launch" \
  PREFILL_MAX_RUNNING=64 \
  PREFILL_GRAPH_MAX_BS=64 \
  PREFILL_HICACHE=0 \
  DECODE_MAX_RUNNING=64 \
  DECODE_GRAPH_MAX_BS=64 \
  DECODE_HICACHE=0 \
  DECODE_SIMULATE_ACC_LEN=3.61 \
  AGENTX_WARMUP_REQUESTS_PER_LANE=1

cleanup() {
  if [[ ! -e "$SHARED_OUT/stop" ]]; then
    ./stop.sh \
      TOPOLOGY=topology.2p1d.tsv \
      OUT_DIR="$SHARED_OUT/stop"
  fi
}
trap cleanup EXIT

# 在 control node 执行 runner，使 OUT_DIR、uv/HF cache、逐请求 JSONL 和
# server metrics 都落在 /mnt/m2m_nobackup，而不是共享 /home。
set -o pipefail
ssh -o BatchMode=yes -o ConnectTimeout=10 "$CONTROL_NODE" \
  bash -s -- "$BENCH_DIR" "$REMOTE_OUT" "$REMOTE_CACHE" <<'REMOTE' \
  2>&1 | tee "$SHARED_OUT/agentx-console.log"
set -euo pipefail
bench_dir="$1"
remote_out="$2"
remote_cache="$3"
cd "$bench_dir"
./agentx_bench.sh \
  TOPOLOGY=topology.2p1d.tsv \
  CONC=8 \
  DURATION=1200 \
  OUT_DIR="$remote_out" \
  AGENTX_CACHE_DIR="$remote_cache" \
  AGENTX_WARMUP_REQUESTS_PER_LANE=1 \
  PREFILL_MAX_RUNNING=64 \
  PREFILL_GRAPH_MAX_BS=64 \
  PREFILL_HICACHE=0 \
  DECODE_MAX_RUNNING=64 \
  DECODE_GRAPH_MAX_BS=64 \
  DECODE_HICACHE=0 \
  DECODE_SIMULATE_ACC_LEN=3.61
REMOTE

# 保留 control node 上的完整原始数据，只复制聚合结果和小型审计文件。
mkdir -p "$SHARED_OUT/agentx-c8"
scp "$CONTROL_NODE:$REMOTE_OUT/agentx_conc8.json" \
  "$CONTROL_NODE:$REMOTE_OUT/runtime.env" \
  "$CONTROL_NODE:$REMOTE_OUT/runner.log" \
  "$CONTROL_NODE:$REMOTE_OUT/benchmark.log" \
  "$CONTROL_NODE:$REMOTE_OUT/benchmark_command.txt" \
  "$SHARED_OUT/agentx-c8/"
scp -r "$CONTROL_NODE:$REMOTE_OUT/service" "$SHARED_OUT/agentx-c8/"
printf '%s\n' "$CONTROL_NODE:$REMOTE_OUT" \
  > "$SHARED_OUT/agentx-c8/REMOTE_ARTIFACTS.txt"

test -s "$SHARED_OUT/agentx-c8/agentx_conc8.json"
python3 -m json.tool \
  "$SHARED_OUT/agentx-c8/agentx_conc8.json" >/dev/null

cleanup
trap - EXIT
```
