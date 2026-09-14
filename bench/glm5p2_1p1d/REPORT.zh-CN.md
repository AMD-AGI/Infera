# GLM-5.2 Agentic 1P1D 实验报告

状态：已完成  
工作区版本：`8f04a68cc5c33aa0f8dae69615aad09a8ee36fd6`

## 目标与验收标准

通过 Infera/SGLang，以一个预填充（prefill）节点加一个解码（decode）节点的方式部署
`GLM-5.2-MXFP4`，并满足以下要求：

1. 正确性：一个预填充 worker 和一个解码 worker 持续保持注册；普通对话语义连贯；
   GLM 工具解析器能够生成有效的 OpenAI 函数调用。
2. 传输：容器内可见 RDMA 设备；Mooncake 不报告 TCP 回退或空 GID；
   跨节点带宽探测成功。
3. 功能状态：预填充和解码端解析为预期的 DP-attention 形态；
   EAGLE 能提供接受长度证据，且不发生 DP 死锁。
4. 并发：固定形态下，8、16、32、64 和 128 并发的扫描均完成，且请求失败数、
   worker 丢失、调度器 traceback 和请求撤回均为零。
5. Agentic 稳定性：运行 AIPerf `inferencex-agentx-mvp`，并发 14，持续 3600 秒，
   固定使用 `semianalysis_cc_traces_weka_062126`，请求失败数为零。
6. 优化：每组安全的 PR 都与同一基线进行对比测量，而不是一次性启用所有尚未合并的改动。

## 执行计划

### 阶段 A —— 环境与镜像

- 选择两个空闲节点，并记录 GPU/容器使用情况。
- 探测 peer-memory、ODP、GID 以及容器内 provider 的能力。
- 构建当前检出版本中的 `deploy/docker/Dockerfile.sglang`。
- 使用该准确镜像重新执行注册模式和跨节点网络探测。

### 阶段 B —— 功能基线

- 启动 etcd。
- 通过安全的 RDMA 模式启动预填充端和解码端。
- 两个 worker 均健康后，启动 Infera 路由器。
- 执行对话、工具调用、worker、日志和功能检查。

### 阶段 C —— 负载基线

- 执行固定 8K 输入/1K 输出的并发扫描。
- 执行一小时 AgentX 工作负载。
- 保存逐请求产物和引擎日志。

### 阶段 D —— 优化 A/B 测试

- 将 `glm5.2_single_opt.md` 中的每个 PR 分类为正确性、兼容性或性能改动。
- 构建依赖完整的最小镜像组合。
- 每组镜像依次重新执行冒烟测试和有代表性的合成测试。
- 只有通过较短门禁测试的组合才运行完整 AgentX 测试。

## 环境调查结果

2026-09-02 17:13（UTC+8）的节点扫描结果：

- `crsuse2-m2m-136` 和 `crsuse2-m2m-140` 最初处于空闲状态，用于镜像构建和首次网络探测。
- `135`、`138` 和 `139` 上有活跃的 SGLang/VLLM 任务，因此未选用。
- 首次网络探测期间，无关的 `glm52_pd_ab` 工作负载在 136 和 140 上启动，
  占用了所有 GPU 约 90% 的显存。实验仅停止了自身挂起的预检容器，未触碰这些工作负载。
- 重新扫描时，`crsuse2-m2m-137` 和 `crsuse2-m2m-138` 已完全空闲，
  每个节点根磁盘约有 89 GB 可用空间。它们被选为最终的预填充和解码节点。
- 重新扫描时 `142` 也处于空闲状态，保留为备用节点。
- `141` 的 GPU 同样空闲，但根磁盘仅剩约 21 GB；为避免在基线构建前删除无关镜像缓存，
  未选用该节点。
- 两个选定节点均提供八个处于活动状态、速率为 400 Gb/s 的 `ionic_0..7` 端口，
  以及一个速率为 200 Gb/s 的活动 `mlx5_0` 端口。
- 两者均使用 `amdgpu-dkms 1:6.14.14.30100100-2212064.24.04`；
  宿主机 `rocm-core` 版本为 7.0.1。
- 模型位于 `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`。

## 注册模式调查结果

初始硬件探测仅借用现有镜像作为诊断载体；之后会使用新构建的实验镜像重复该探测。

最初选定的节点报告了相同的能力；最终节点对将使用最终镜像再次探测：

- 8 × gfx950 GPU。
- `CONFIG_PCI_P2PDMA=y`。
- 无 peer-memory 模块。
- Mooncake 构建时包含 `ibv_reg_dmabuf_mr`。
- Ionic 链路支持 dma-buf，但不提供 ODP。
- `mlx5_0` 同时支持 dma-buf 和 ODP。

由此得出：

- 由于缺少 peer-memory，无法在所有链路上直接使用 `ibv_reg_mr`。
- 安全的免 pin 模式是在 `mlx5_0` 上使用 dma-buf，并设置
  `MC_GID_INDEX=3`、`MC_MS_FILTERS=mlx5_0`、`MC_MS_AUTO_DISC=0` 和
  `MOONCAKE_DISABLE_HIP_DMABUF=0`。
- 该安全模式会将可用的 KV 传输链路限制在 200 Gb/s。
- 多链路 Ionic 仅适用于受限的模式 C。注册会 pin 并复制 KV 池，
  因此必须先设置 GLM-5.2 专用的 token 上限，才能安全启动。
  该方案将在基线完成后再考虑。

## 镜像调查结果

仓库中的 Dockerfile 当前固定使用 SGLang `v0.5.17-rocm720-mi35x`，并应用：

- 重新构建的 Mooncake，支持跨主机路由和 dma-buf；
- PD + DP-attention + EAGLE 所需的四个 GLM-5.2 DSA 补丁；
- Mooncake 分块预填充提前发送的正确性修复；
- ROCm HiCache 主机内存分配修复；
- 当前 Infera 源码和 Rust 路由器。

初始基线镜像构建：

- 标签：`infera/engine-sglang:glm52-1p1d-8f04a68`
- 构建主机：`crsuse2-m2m-136`、`crsuse2-m2m-140`
- 状态：两个节点均构建成功。

首次完整网络探测完成了 9×9 的 `ib_write_bw` 矩阵，但 Mooncake 包装器使用严格 UTF-8
解码子进程的非 UTF-8 日志时失败。Rank 0 抛出 `UnicodeDecodeError`，随后 Rank 1
持续等待对端。该次运行已停止，不采用其中的带宽结果。

下一次运行暴露了第二个探测假设：程序从 `ionic_0` 读取 GID 1，并将其复用于包括
`mlx5_0` 在内的所有设备，而 `mlx5_0` 要求使用 GID 3。预检现已支持显式设备过滤，
在过滤后获取 GID，并将同一过滤条件传入 Mooncake 的 GPU 测试。针对输出解码、
设备过滤和降级链路检测的七个专项单元测试均已通过。

最终实验镜像：

- 标签：`infera/engine-sglang:glm52-1p1d-8f04a68-pf2`
- 构建主机：`crsuse2-m2m-137`、`crsuse2-m2m-138`
- 状态：两个节点均构建成功。

分层 A/B 镜像：

- `glm52-1p1d-8f04a68-corr-1` 仅应用
  [SGLang #37133](https://github.com/sgl-project/sglang/pull/37133)，
  在参数加载阶段和 Aiter 边界处，将 GLM-5.2 的 MoE 修正偏置保留为 fp32。
  上游测量显示，bf16 会将 238 个偏置值压缩为 8 个，并改变 98.5% token
  所选择的 top-8 expert 集合。这是新增分层中唯一启用的语义正确性修复。
- `glm52-1p1d-8f04a68-runtime-1` 额外加入
  [SGLang #37118](https://github.com/sgl-project/sglang/pull/37118)，
  在 HIP 上开放纯 torch DSA head-gate 的图辅助函数。该 PD 配置会自动禁用预填充图，
  但解码端会捕获完整的 EAGLE target-verify 图和 draft 图，因此这个已合并的图兼容性改动
  保留在通过验收的 runtime 分层中。
- `glm52-1p1d-8f04a68-opt-1` 还包含
  [Aiter #5121](https://github.com/ROCm/aiter/pull/5121) 中无额外依赖的
  `fp8_mqa_logits` offset-gate 代码片段。源码核验表明，当前固定的旧版 Aiter
  不含后来加入的 `BLOCK_M=2` 中止逻辑，并且对大缓冲区已经会回退到全局 load/store。
  不含该代码片段的接近上限运行时测试同样通过，因此 `opt-1` 仅保留为历史实验分支，
  不再作为默认镜像。
- 所有分层均以本地构建的 `pf2` 镜像为父镜像。构建时的 `git apply --check`、
  失败即终止的源码锚点替换和 `compileall` 在两个节点上均通过。
- [SGLang #37124](https://github.com/sgl-project/sglang/pull/37124)、
  [#37134](https://github.com/sgl-project/sglang/pull/37134) 和
  [Aiter #5122](https://github.com/ROCm/aiter/pull/5122)
  需要更新且依赖完整的 kernel 与调用方布局，因此未做局部应用。
  当前 Infera 启动屏障只覆盖
  [SGLang #37152](https://github.com/sgl-project/sglang/pull/37152)
  中的协同调度部分，其余部分未做向后移植。HiCache 仍处于禁用状态。
  [SGLang #37130](https://github.com/sgl-project/sglang/pull/37130)
  没有独立的上游 diff，而
  [#37117](https://github.com/sgl-project/sglang/pull/37117) 已关闭且没有代码。

修正后的专项网络测试：

- 两个方向：`ib_write_bw` 1/1 链路均通过 `mlx5_0`、GID 3 成功。
- 138 → 137 带宽：21.78 GB/s。
- 137 → 138 带宽：6.56 GB/s。
- Mooncake 主机内存 RDMA：5.12–5.52 GB/s；TCP 控制路径：0.08 GB/s。
- Mooncake 真实显存传输：两个方向上的全部 8 个 GPU 均通过字节校验，
  带宽为 16.72–18.01 GB/s。
- 产物：
  `results/preflight_fabric_targeted_20260902_102312/infera_preflight_report.html`。

## 结果

### 基线部署

启动耗时 8 分 44 秒：

- 预填充端：`10.245.153.247:30001`，TP8/EP8，解析后的 DP size 为 1，
  DP-attention 为 false。
- 解码端：`10.245.157.237:30002`，TP8/EP8/DP8，解析后的 DP size 为 8，
  DP-attention 为 true。
- 路由器：`10.245.153.247:8000`，使用 `kv-aware`，有两个活跃 worker，
  KV block size 为 64。
- 每个 Mooncake worker 均在 `mlx5_0`、GID 3 上安装了 RDMA 传输。
  解码日志中包含真实路由请求发出的 RDMA 就绪确认。

冒烟测试：

- 普通对话返回了正确句子 `The capital of France is Paris.`。
- 强制选择工具时，返回了有效的 OpenAI 函数调用
  `get_weather({"city":"Paris"})`。
- `MC_FORCE_TCP`、`GID is NULL`、`KVTransferError`、DSA 行不匹配、
  Python traceback 或 GPU 内存故障日志特征均为零。
- EAGLE 接受长度：3 个样本，中位数 2.98，范围 2.67–3.16。
- 产物：`results/smoke_20260902_104135/`。

### 基线合成负载

8K 输入 / 1K 输出，并发 8，共 80 个计量请求：

- 80/80 成功，耗时 131.63 秒。
- 输出吞吐量：622.35 token/s；总吞吐量：5,601.19 token/s。
- TTFT 中位数：498.61 ms；P99 TTFT：9,369.76 ms。
- TPOT 中位数：11.08 ms；P99 TPOT：13.38 ms。
- ITL 中位数：27.56 ms；P99 ITL：32.74 ms。
- 该档测试后无 worker 丢失。
- 产物：`results/synthetic_baseline_c8_retry_20260902_104259/`。

其余基线并发档均完成：

- C16：160/160 成功，耗时 148.83 s；输出吞吐量 1,100.82 tok/s；
  TTFT 中位数/P99 为 512.53/12,901.30 ms；TPOT 中位数/P99 为
  12.32/15.07 ms；缓存命中率 49.7%。
- C32：320/320 成功，耗时 157.74 s；输出吞吐量 2,077.34 tok/s；
  TTFT 中位数/P99 为 517.07/3,266.43 ms；TPOT 中位数/P99 为
  14.12/17.43 ms；缓存命中率 49.7%。
- C64：640/640 成功，耗时 215.56 s；输出吞吐量 3,040.29 tok/s；
  TTFT 中位数/P99 为 1,615.98/11,423.11 ms；TPOT 中位数/P99 为
  16.41/20.49 ms；缓存命中率 50.1%。
- C128：1,280/1,280 成功，耗时 489.95 s；输出吞吐量 2,675.19 tok/s；
  TTFT 中位数/P99 为 30,814.53/43,133.58 ms；TPOT 中位数/P99 为
  16.17/19.84 ms；缓存命中率 0.5%。

五个并发档合计 2,480/2,480 个计量请求成功。扫描后的冒烟测试仍检测到两个活跃 worker，
对话与工具调用输出有效，未发现 Mooncake/DSA/GPU 故障特征，且非零
`#retracted-req` 为零。解码日志中包含 6,074 个 EAGLE 接受长度样本
（中位数 2.50，范围 1.57–3.81）。C64 是吞吐量拐点：C128 会显著增加预填充排队，
使输出吞吐量下降 12.0%，并将 TTFT 中位数提高到 30.8 秒。

- 扫描产物：`results/synthetic_baseline_sweep_20260902_104653/`。
- 扫描后产物：`results/smoke_post_sweep_20260902_110656/`。

### 优化门禁

在完整停止旧部署后，优化镜像启动成功，并通过了相同的 worker、对话、工具调用、
Mooncake、DP-attention 和 MTP 检查。首次原地重启时，Infera 自动分配的 KV-event
ZMQ endpoint 出现了瞬时端口释放竞争（`Address already in use`）。进程退出后没有
残留 listener；完整停止后重试成功。`launch.sh` 现在会同时清理旧路由器和两端实例，
并等待五秒后再启动新部署。

当前的 [SGLang #37134](https://github.com/sgl-project/sglang/pull/37134)
采样修复不能安全地单独移植到 v0.5.17 代码树：它新增了 Triton rejection sampler
以及参数解析改动，并且上游尚未合并。缺少该修复时，ROCm EAGLE verify 会在非贪心请求上
静默使用 argmax。因此，基准测试脚本将 temperature 固定为 0，使请求的采样约定与实际
执行路径保持一致。

- 冒烟测试产物：`results/smoke_optimized_smoke_20260902_112320/`。

优化后的合成测试结果：

- C16 冷缓存门禁：160/160 成功，耗时 146.10 s；输出吞吐量 1,121.44 tok/s。
  这不能与基线直接比较，因为基线 C16 从之前的 C8 档继承了约一半 prompt prefix。
- C32 首次运行：320/320 成功，耗时 174.91 s；输出吞吐量 1,873.43 tok/s。
- C64：640/640 成功，耗时 217.02 s；输出吞吐量 3,019.82 tok/s；
  TTFT 中位数/P99 为 1,613.36/10,143.54 ms；TPOT 中位数/P99 为
  16.88/20.37 ms。
- C128：1,280/1,280 成功，耗时 489.07 s；输出吞吐量 2,680.02 tok/s；
  TTFT 中位数/P99 为 30,328.32/43,156.66 ms；TPOT 中位数/P99 为
  16.50/20.38 ms。
- 一次控制缓存状态的 C16→C32 重复测试中，C32 的 320/320 个请求成功，
  耗时 163.04 s，输出吞吐量为 2,009.85 tok/s，缓存命中率为 49.7%。

首次扫描的四个并发档均在包装器返回状态码 2 前写出了完整指标：
进程已经解析 case 分支的一部分后，`bench.sh` 被在线编辑，旧 shell 直到 C128
完成后才遇到变更后的文本。最终文件通过 `bash -n`，随后 C16→C32 调用以状态码零退出；
这不是引擎或请求失败。

在缓存行为一致的两个饱和点，输出吞吐量在 C64 变化 -0.7%，在 C128 变化 +0.2%。
C32 重复测试为 -3.2%。TPOT 中位数上升约 1.6–2.9%，方向上与 fp32 路由开销一致。
因此，组合镜像验证的是兼容性和正确性，而不是 8K 上下文的吞吐量收益。
下述后续隔离测试进一步缩小了生产镜像的组成范围，并移除了不必要的 Aiter 代码片段。

在 2,880 个计量优化请求完成后，扫描后的冒烟测试仍通过：两个 worker 保持活跃，
所有检索的致命错误特征均为零，7,079 个 EAGLE 样本的接受长度中位数为 2.49
（范围 1.77–4.00）。

- 扫描产物：`results/synthetic_optimized_sweep_20260902_112416/`。
- 受控重复测试产物：`results/synthetic_optimized_repeat_20260902_114435/`。
- 扫描后产物：`results/smoke_optimized_post_sweep_20260902_115107/`。

### 后续补丁隔离测试

最终的 PR/源码映射显示，组合后的 `opt-1` 分层超出了当前固定运行时的需求，因此在两个节点上
分别独立构建了仅含 #37133 的镜像，以及包含 #37118+#37133 的镜像。

仅含 #37133 的 `corr-1` 镜像通过了 worker、对话、工具调用、Mooncake、
DP-attention 和 MTP 冒烟检查。控制缓存状态的 C32→C64 测试完成了 960/960 个请求：

- C32 冷缓存：输出吞吐量 1,946.70 tok/s；TPOT 中位数/P99 为 13.87/17.25 ms。
- C64 的继承缓存状态与原始扫描同为 50.1%：输出吞吐量 2,905.81 tok/s；
  TPOT 中位数/P99 为 16.37/20.36 ms。

`runtime-1` 镜像加入 #37118、但不包含 Aiter #5121。该镜像通过了相同的冒烟检查，
并再次完成了 960/960 个请求：

- C32 冷缓存：输出吞吐量 1,831.03 tok/s；TPOT 中位数/P99 为 14.45/18.31 ms。
- C64 的缓存命中率为 50.1%：输出吞吐量 2,820.84 tok/s；
  TPOT 中位数/P99 为 16.75/20.61 ms。

这些后续测试在原始 A/B 测试数小时后执行，呈现了更高的预填充/TTFT 方差；
它们不能证明 #37118 带来了吞吐量提升。解码 TPOT 仍处于相同范围，且所有请求均成功。
因此保留 #37118 是为了支持活跃的解码图接口，而不是将其作为性能优化宣传。

不含 Aiter #5121 代码片段时，`runtime-1` 在 4.78 秒内完成了一个 70,017-token 请求，
并在 24.16 秒内完成了一个 250,016-token 请求。AgentX 之后可复现运行的
`long_context.sh` 在 24.83 秒内重复完成了 250,016-token 用例。
这验证了当前固定 Aiter 在接近所配置 262,144-token 上限时的回退路径，
并将 #5121 从生产要求中移除。

- 仅 #37133 冒烟测试：`results/smoke_corr_only_smoke_20260902_2244/`。
- 仅 #37133 的 C32→C64：`results/synthetic_corr_only_c32_c64_20260902_2251/`。
- Runtime 冒烟测试：`results/smoke_runtime_smoke_20260902_2314/`。
- Runtime C32→C64：`results/synthetic_runtime_c32_c64_20260902_2315/`。
- Runtime 最终冒烟测试：`results/smoke_runtime_final_20260902_2333/`。
- 可复现的接近上限请求：
  `results/long_context_runtime_post_agentx_20260903_0003/`。

### AgentX-MVP

AIPerf 0.12 接受了 `inferencex-agentx-mvp` 场景锁定配置，强制启用流式传输，
注入 `ignore_eos=true`，选择首轮 prefix 缓存扰动，并将该场景的全局最大空闲间隔设置为
10 秒。在固定使用的 393 条 trace 数据集中，220 条符合 262,144-token 上下文上限，
重建后得到 1,393 个对话。

快照预热完成了 16/16 个请求，无错误或取消，耗时 250.86 秒。
其全局空闲时间上限跳过了记录中的 37,474.64 秒空闲时间，且未改变请求顺序。

C14、3,600 秒的 profiling 窗口共发送 785 个请求。785 条记录导出均存在，
并报告 `was_cancelled=false`、无错误、无上下文溢出跳过、无 OSL 不匹配：

- 输入 68,984,000 token，输出 633,562 token。
- 0.2173 request/s，输出吞吐量 175.35 token/s。
- TTFT 中位数/P99：764.36/4,868.80 ms。
- 请求延迟中位数/P99：3,674.62/64,389.31 ms；最大值 241.74 s。
- ITL 中位数/P99：11.79/17.79 ms。
- 报告的 prompt 缓存读取比例：95.25%。
- 实测平均请求并发为 1.94（最大 6），低于 C14 lane 目标；
  原因是该场景保留了每条 trajectory 中记录的思考时间和 join 依赖。

AIPerf 0.12 未自动完成本次运行。最终响应记录在停止发送 15 秒后导出，
当时已无剩余基准测试 TCP 连接或引擎请求，但系统仍在跟踪一个尚有延迟子节点未完成的
未来 DAG join。设置 `--benchmark-grace-period inf` 后，这条不经过网络的分支使完成事件
无限期保持 pending；即使 1,800 秒 HTTP timeout 也与此无关。系统静默 59 分钟后，
控制器被正常停止以强制导出聚合结果。控制器以状态码零退出并汇总了全部 785 条记录，
但正确地将 `submission_valid=false`，原因为 `run_cancelled`。

这是 AIPerf 控制面最终收尾问题，不是模型请求失败：清理过程报告了一个未来 join，
且无分支错误；与此同时，两个引擎上的运行中或排队请求均为零。
`bench.sh` 现在默认使用有限的 600 秒 grace period，使延迟分支的记账等待时间有上界。

900 秒最短持续时间的对照测试确认了这一行为：

- 127/127 个请求完成；取消请求、请求错误、上下文溢出跳过或 OSL 不匹配均为零。
- 停止发送时存在的两个网络请求均在 21 秒内完成。
- 同一个未来 join 仍然存在，因此 runner 用满了 600 秒 grace period；
  随后确认 `in_flight=0`，并在不取消请求的情况下完成清理。
- AIPerf 以状态码零退出，`was_cancelled=false` 且 `submission_valid=true`。
- 输入 12,885,686 token，输出 132,630 token；输出吞吐量 88.42 token/s。
- TTFT 中位数/P99：853.27/5,221.58 ms。
- 请求延迟中位数/P99：5,942.53/85,609.32 ms。
- 报告的 prompt 缓存读取比例：95.64%。

运行后的冒烟测试通过：两个 worker 均保持活跃，对话和工具调用响应有效，
所检索的 Mooncake/DSA/GPU 故障特征均为零；12,442 个 EAGLE 样本的接受长度中位数为
2.55（范围 1.68–4.00）。对照测试后的最终冒烟测试同样通过；此时累计解码日志包含
13,532 个 EAGLE 样本，接受长度中位数为 2.56。

- 完整时长产物：`results/agentic_optimized_agentx_20260902_115144/`。
- 场景有效的对照测试产物：
  `results/agentic_optimized_agentx_control_20260902_140027/`。
- 运行后冒烟测试产物：`results/smoke_optimized_post_agentx_20260902_140007/`。
- 最终冒烟测试产物：`results/smoke_optimized_final_20260902_143033/`。

随后，已验收的 `runtime-1` 镜像也单独进行了 900 秒、场景有效的对照测试：

- 预热完成 16/16 个请求，无错误或取消。
- Profiling 完成 131/131 个请求；导出的全部 147 条预热加 profiling 记录均报告：
  无取消、请求错误、上下文溢出跳过或 OSL 不匹配。
- 输入 13,139,698 token，输出 134,780 token；输出吞吐量 89.85 tok/s。
- TTFT 中位数/P99：677.01/5,216.65 ms。
- 请求延迟中位数/P99：6,389.90/75,964.04 ms；最大值 244.12 s。
- ITL 中位数/P99：11.13/16.56 ms。
- 报告的 prompt 缓存读取比例：95.21%。
- 同一个未来 join 用满了 600 秒 grace period，但全部 131 个网络请求此前均已完成。
  AIPerf 以状态码零退出，`was_cancelled=false`，错误和取消数均为零，
  且 `submission_valid=true`。

随后的接近上限请求和最终冒烟测试均通过。两个 worker 均保持活跃，
所有检索的 Mooncake/DSA/GPU 故障特征均为零，解码日志中包含 3,094 个 EAGLE 样本，
接受长度中位数为 2.54（范围 1.68–4.00）。

- Runtime AgentX 对照测试：
  `results/agentic_runtime_agentx_control_20260902_2335/`。
- Runtime AgentX 后冒烟测试：
  `results/smoke_runtime_post_agentx_20260903_0003/`。

## 最终结论

该 1P1D 部署通过了所测试贪心 Agentic 工作负载的验收。应使用
`infera/engine-sglang:glm52-1p1d-8f04a68-runtime-1`，保持预填充端
DP-attention 关闭、解码端 DP-attention 为 DP8，并在这些节点上继续使用
`mlx5_0`/GID 3 作为安全的 Mooncake 免 pin 路径。C64 仍是合成负载的吞吐量拐点；
C128 虽然稳定，但会显著增加 TTFT，且不能提升吞吐量。

#37133 是必需的路由正确性向后移植。由于解码端 EAGLE 图处于启用状态，#37118
作为已合并的 HIP 图兼容层予以保留，但这些测试未证明其带来吞吐量提升。
Aiter #5121 不纳入默认镜像：当前固定源码中不含后来加入的中止逻辑，
且不应用该代码片段时通过了 250,016-token 回退测试。`opt-1` 仅保留用于复现历史 A/B 测试。

一小时 AgentX 负载在 `opt-1` 上运行；缩减后的 `runtime-1` 镜像另外通过了有效的
900 秒对照测试、接近上下文上限的测试以及运行后冒烟测试，未发生模型或传输失败。
在集成并重新验证更新的 rejection-sampling 技术栈之前，非零 temperature 的 EAGLE
采样仍不在验收范围内。AIPerf 0.12 的延迟 join 收尾问题仍然存在，但有限的 grace 设置
可使运行时间有界，并生成有效提交，且网络请求的失败或取消数均为零。
