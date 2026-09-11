# GLM-5.2 fake-decode c32 profiling

在 `packups/glm52_fake_tp4ep4_10k500_c16_c32.packup_20260910-055810` 的服务端与 benchmark 基础上采集 CPU/GPU trace，分析 conc=32 的 GPU 算子分布。原始 packup 和历史证据不改动。

本次节点测试的数值、分析和 trace 入口见 [REPORT.md](REPORT.md)。

保留 TP4/EP4/DP1、GPU 0–3、FP8 KV、mem-fraction=0.85、ISL=10000、OSL=500、EAGLE steps=5 / draft tokens=6 / topk=1、模拟接受长度 3.61、原有三份 fake-decode 补丁、CUDA Graph 与 attention fallback。每轮仍是 16 个 warmup 请求与 128 个测量请求；原生客户端 warmup 的输出长度为 32，测量请求输出长度为 500。

**ROCm 7.2 采集修正：**默认设置 `DEBUG_CLR_GRAPH_PACKET_CAPTURE=false`，关闭 ROCm 图的 packet capture 优化，确保 target verify 图内 kernel 可见；CUDA Graph 仍开启。实测原模式的大图 trace 会漏记 kernel。这个开关改变图回放开销，因此完整 trace 用于分析算子构成，不能将其耗时比例直接当成原优化模式的精确延迟分解。原始性能需在独立运行目录使用 `prepare.py --hip-graph-mode original`，启动后运行 `collect.py --baseline-only`。

## 脚本

- `prepare.py`：调用 packup 的 relocation helper 创建全新运行目录，核对镜像和 SGLang/AITER commit，适配 host 模型路径和节点本地 cache，保存启动差异及环境信息。默认要求历史镜像 ID；同 commit 的其他镜像需明确传入 `--allow-image-mismatch`，报告会标记差异。
- `collect.py`：等待服务 ready，先测一轮本模式无 profiler 基准，再抓两轮 trace。连续两条 scheduler 日志都满足 running=32、retracted=0、CUDA Graph=True 后，调用原生 `/start_profile`，默认抓 12 个 forward 并自动停止。每轮保留四个 TP rank 的 `.trace.json.gz`，检查 128/128 成功、实际输入/输出长度、verify batch=32，以及每次图回放都有 GPU kernel。
- `analyze.py`：标准库离线分析，输出类别、kernel、阶段、逐 rank CSV、逐轮分布、`summary.json` 和 `SUMMARY.md`；默认拒绝有图回放却没有 kernel 的 trace。用 `--kernel-map kernel_aliases.json` 加载已核实的本模型匿名 kernel 映射。
- `plot.py`：用镜像内已有的 matplotlib 输出独立 PNG/SVG 分布图。
- `test_analyze.py`：验证重叠 kernel、异步 graph attribution、kernel 时间分母和 memcpy 排除。
- `graph_probe.py`：在同镜像内交替回放大小两个图，应捕获 3603 个 kernel；可独立复现 packet capture 导致的漏记。

## 在指定节点复现

先连接 `ssh smci355-ccs-aus-n04-33`，在仓库根目录运行。以下目录与容器名必须使用新值；原 launcher 会检查八张 GPU 的显存占用均不高于 2%，并拒绝替换已有容器。

```bash
ROOT="$PWD/examples/glm52_fake_tp4ep4_profile"
STAMP=$(date -u +%Y%m%dT%H%M%S)
RUN="$ROOT/.cache/run_$STAMP"
NAME="glm52-profile-$STAMP"
mkdir -p "$ROOT/.cache"
python3 "$ROOT/prepare.py" "$RUN" \
  --model /data/models/GLM-5.2-MXFP4 \
  --image sha256:416d51effc431e27a4ddeed5cdd8ca6012c68eeabb071f987ce99f1ba9854973 \
  --allow-image-mismatch \
  --cache-dir "/data/cyao1002/glm52_profile_$STAMP" \
  --name "$NAME" --port 31832
bash "$RUN/scripts/launch.sh" sim c32 0.85 32
python3 "$ROOT/collect.py" "$RUN" \
  --output "$ROOT/results/$STAMP" --steps 12 --repeats 2
python3 "$ROOT/analyze.py" "$ROOT/results/$STAMP" \
  --kernel-map "$ROOT/kernel_aliases.json" \
  --output "$ROOT/results/$STAMP/analysis"
docker stop --timeout 300 "$NAME"
docker inspect "$NAME" --format '{{.State.Running}} {{.State.ExitCode}}'
```

`collect.py` 不停止服务；失败时保留现场供检查。只停止本次创建的容器。冷 cache 需要等待模型加载及 JIT 编译，可用 `docker logs --tail 30 "$NAME"` 查看进度。

## 阅读结果

将 trace 解压后拖入 [Perfetto](https://ui.perfetto.dev)，可逐 rank 查看 `draft`、`step[TARGET_VERIFY bs=32]`、`draft_extend`、CPU launch 与 GPU kernel。采集使用 `DEBUG_CLR_GRAPH_PACKET_CAPTURE=false` 保留图内 GPU kernel，关闭 Python stack 和 shape 记录以控制采集开销。此 ROCm runtime 必须使用布尔字面量 `false`；实测 `0` 没有关闭 packet capture。`TORCH_PROFILER_HIP_GRAPH_TRACING` 在该 PyTorch 构建中不能解决漏记。

类别占比的分母是所选 trace/rank 中 **GPU kernel duration 的累加值**，不是请求延迟，也不是四卡实例的墙钟时间。重叠 kernel 会分别计时；`summary.json` 另列每卡 kernel 区间并集和首尾跨度。CPU launch/annotation 及 memcpy 不计入 kernel 时间分母。

算子分类使用 `analyze.py` 中有序的 kernel 名称规则；融合 allreduce+norm 的 kernel 整体计入通信类别，无法从一个 kernel 拆出内部各算子时间。逐 kernel CSV 保留完整名称，便于审核归类。MTP 阶段通过 GPU 与 CPU launch 的 correlation/external ID 关联；无法关联的保留为 `unattributed`，不会按 CPU/GPU 时间重叠猜测异步归属。

`kernel_aliases.json` 仅适用于这里固定的模型/commit。`main_kernel` 经源码与全部 verify 图中的相邻 kernel 验证，对应 TileLang sparse MLA 的 partial/combine；`p0v2_kernel_0`、`p23_kernel_1` 对应 FlyDSL MoE sorting。换模型或实现后需重新核实，不应复用泛化命名推断。

profile 轮吞吐包含采集和 trace 导出开销；`baseline_c32/metrics.json` 是**该运行模式**的无 profiler 性能参考，需同时查 `state/profile-environment.json` 中的 `hip_graph_mode`。这仍是 synthetic KV / simulated acceptance 测试，不代表真实 prefill 或 PD 正确性。C32 verify=32×6=192 行，超过 FlyDSL sparse MLA 的 96 行上限；必须结合实际日志和 kernel 名称解释 fallback。

```bash
python3 -m unittest discover -s examples/glm52_fake_tp4ep4_profile -p 'test_*.py'
```
