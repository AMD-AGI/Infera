# GLM-5.2-MXFP4 P4+DPA / D8+DPA AgentX 实验报告

## 实验目标与固定口径

- 模型：`GLM-5.2-MXFP4`
- 部署：1 Prefill + 1 Decode
- Prefill：TP4 / EP1 / attention-DP4，GPU 0-3
- Decode：TP8 / EP1 / attention-DP8，GPU 0-7
- HiCache：Prefill、Decode 均关闭
- Decode MTP：EAGLE，5 steps / 6 draft tokens / top-k 1
- Correctness：真实 MTP 接受，`DECODE_SIMULATE_ACC_LEN=`
- AgentX：固定模拟接受长度 `3.61`
- AgentX 并发：8、16、32、48、64
- 每个 AgentX 点的统计窗口：1200 秒
- 每次 concurrency 改变都执行完整 stop → HBM idle → launch
- 每点 Prefill/Decode 的 `max_running_requests` 和 `cuda_graph_max_bs` 都等于该点 concurrency
- 不跨点复用 prefix cache

完成状态：Correctness、C8、C16、C32 已完成；C48 经基线及两个单变量显存 A/B
均确认超过 P4 Prefill 容量，C64 因相同且更高的资源需求记为不可运行。有效性能汇总
包含三个点，失败点保留完整现象、日志、debug 和处理记录。

### 瓶颈判定口径

每个并发点除 AgentX 汇总值外，还读取
`aiperf_artifacts/server_metrics_export.{json,csv}`。多 DP rank 的 histogram 按
`count/sum/buckets` 合并；不直接平均各 rank 的 P90，因为那不是全局 P90。

- Prefill 瓶颈：Prefill scheduler queue 明显堆积，且 TTFT 同步变差，而 Decode queue
  仍低。
- Decode 瓶颈：Decode running requests 接近并发上限、Decode queue 持续堆积，ITL
  变差且总吞吐增益开始收敛。
- KV transfer 瓶颈：Decode transfer/prealloc queue 持续堆积，或 transfer latency
  在 TTFT 中占比显著且速度持续下降，同时 P/D compute queue 不能解释该现象。
- 未饱和：两端 queue 均低、吞吐仍近似随 concurrency 增长。此时仅比较哪一端更忙，
  不强行指定瓶颈。

判断同时考虑 cache hit。高 prefix hit 会显著降低实际 Prefill token 计算量，因此不能
只根据输入 token/s 或 P/D GPU 数量推断瓶颈。

## 节点与镜像

- Prefill：`crsuse2-m2m-137`，`10.245.153.247`
- Decode：`crsuse2-m2m-138`，`10.245.157.237`
- 镜像：`infera/engine-sglang:glm52-v518-c29bd17-b02ab81`
- 两节点实际 Image ID：`sha256:8bef14c08ba4764698b5b9c82f41e1a3e5c0f7bc3c4a22d161cd00dbf1feb456`
- Mooncake：`ionic_0..7`，GID index 1，DMA-BUF
- 本轮隔离端口：etcd client 12379、etcd peer 12380、router 18000

同名镜像在 137/138 上原先指向 `sha256:869fbd...`，与已通过验证的 140/136/139 节点镜像 ID 不一致。为避免将镜像差异混入 A/B，使用现有 `build_image.sh distribute` 从 140 将 `8bef14...` 精确同步到 137/138。

## 基础设施与 Correctness

### 节点及 Mooncake

- 137 的 Prefill GPU 0-3：启动前 HBM 0%
- 138 的 Decode GPU 0-7：启动前 HBM 0%
- 非对称 P4→D8 Mooncake WRITE：PASS
- 字节校验覆盖 4 条 P/D 公共 GPU WRITE 路径

### Correctness 结果

- Smoke：PASS
- 250K long-context：PASS
  - prompt tokens：250,016
  - completion tokens：4
  - elapsed：135.783 秒
- GSM8K 全量 1,319 题：PASS
  - strict-match exact match：96.8916%
  - flexible-extract exact match：97.1190%

Correctness 全程关闭 HiCache，并使用真实 MTP 接受。完成后服务仍健康，随后停止并重新 launch C8 性能栈。后续 C16/C32/C48/C64 不再重复 correctness。

## AgentX C8

- 配置：`CONC=8`
- Prefill/Decode max running：8 / 8
- Prefill/Decode CUDA graph max batch：8 / 8
- 独立 launch 耗时：811.569 秒
- 实际统计窗口：1,224.796 秒
- 计分请求：335
- warmup 请求：87
- 错误请求：0
- 平均 QPS：0.27332
- 总吞吐：32,944.39 tok/s
- 每 GPU 总吞吐：2,745.37 tok/s
- 每 GPU 输入吞吐：2,724.11 tok/s
- 每 GPU 输出吞吐：21.25 tok/s
- P50 TTFT：0.76271 秒
- P90 TTFT：4.26258 秒
- P95 TTFT：7.92405 秒
- P50 E2E：3.82533 秒
- P90 E2E：23.33901 秒
- P95 E2E：32.92467 秒
- P50 ITL：6.66 毫秒
- P90 ITL：7.64 毫秒
- P90 interactivity：130.84 tok/s/user
- GPU cache hit rate：97.573%
- 理论 cache hit rate：97.793%
- GPU KV cache 使用率：27%
- CPU HiCache 使用率：不适用

### C8 瓶颈判断与下一步

结论：**C8 尚未形成稳定的 Prefill、Decode 或 KV transfer 饱和瓶颈；Decode 相对更忙，但当前主要问题是整体负载不足。**

证据：

- Prefill 和 Decode 的 pooled scheduler queue P90 均不超过 5 毫秒，等待队列没有持续堆积。
- Prefill 4 个 DP rank 的平均 running request 合计约 0.10；Decode 8 个 DP rank合计约 5.06。Decode 生命周期明显更长，但每 rank 最大 running request 仍只有 1，未形成持续排队。
- Decode P90 ITL 为 7.64 毫秒，P90 interactivity 为 130.84 tok/s/user，说明 Decode 在该负载下仍保持较好的单用户生成速度。
- Prefill 侧记录 336 次 KV transfer，按 count 加权平均 transfer latency 约 142.03 毫秒、平均速度约 82.30 GB/s；Decode transfer queue P90 为 0，最大仅 1。传输存在尾部抖动，但不是持续吞吐瓶颈。
- GPU cache hit 97.573%，绝大多数长 prompt token 已在 Prefill GPU cache 命中，进一步降低了 Prefill 实算负载。

P/D 比例与优化建议：

- 若目标是本并发点的最低延迟，保持当前 1P4:1D8；增加 Prefill 或 Decode 都缺乏排队证据。
- 若目标是提高集群利用率，不应为 C8 增加 Decode；可以在更大规模部署中让一个 P4 Prefill 服务多个 D8 Decode 副本，或者降低单服务资源，但后者需要另做模型容量与 TP 兼容性验证。
- 不建议仅根据少量 transfer 尾部样本调整 Mooncake 开关。先观察 C16 以上 transfer queue 和速度是否出现持续恶化，再决定是否做网络 A/B。

## AgentX C16

- 配置：`CONC=16`
- Prefill/Decode max running：16 / 16
- Prefill/Decode CUDA graph max batch：16 / 16
- 独立 launch 耗时：823.234 秒
- 实际统计窗口：1,218.951 秒
- 计分请求：602
- warmup 请求：177
- 错误请求：1，`InvalidInferenceResultError`
- 计分错误率：1/603 = 0.166%，低于 10% 门槛
- 平均 QPS：0.49465
- 总吞吐：63,483.52 tok/s
- 每 GPU 总吞吐：5,290.29 tok/s
- 每 GPU 输入吞吐：5,247.54 tok/s
- 每 GPU 输出吞吐：42.75 tok/s
- P50 TTFT：1.48012 秒
- P90 TTFT：5.10527 秒
- P95 TTFT：7.14423 秒
- P50 E2E：5.57345 秒
- P90 E2E：20.52256 秒
- P95 E2E：37.36852 秒
- P50 ITL：8.19 毫秒
- P90 ITL：9.20 毫秒
- P90 interactivity：108.65 tok/s/user
- GPU cache hit rate：96.438%
- 理论 cache hit rate：96.723%
- GPU KV cache 使用率：42%
- CPU HiCache 使用率：不适用

该点通过现有 AgentX 门禁。错误被保留在结果和报告中，没有通过改阈值、过滤数据或增加性能开关来掩盖。

### C16 瓶颈判断与下一步

结论：**C16 已出现以 Prefill 排队为主、KV transfer 退化为辅的 TTFT 瓶颈；Decode 负载上升并拉高 ITL，但尚未出现持续 Decode queue。**

证据：

- Prefill pooled scheduler queue P90 约 2 秒；Decode pooled queue P90 不超过 1 毫秒。两端排队量级差异明确指向 Prefill。
- C8→C16 时 P50 TTFT 从 0.763 秒升至 1.480 秒，P90 TTFT 从 4.263 秒升至 5.105 秒，与 Prefill queue 增长一致。
- Prefill 4 个 DP rank 的平均 running request 合计约 0.52；Decode 8 个 DP rank 合计约 7.65。Decode 更忙，但 scheduler queue 大多数时间为空。
- P90 ITL 从 7.64 毫秒升至 9.20 毫秒，P90 interactivity 从 130.84 降至 108.65 tok/s/user，说明 Decode batching 已带来约 20% 的交互性损失；不过总吞吐仍从 32,944.39 增至 63,483.52 tok/s，尚未证明 Decode 吞吐饱和。
- Prefill 侧记录 603 次 KV transfer，按 count 加权平均 transfer latency 从 C8 的 142.03 毫秒升至 350.88 毫秒，平均速度从 82.30 降至 60.98 GB/s；但 Decode transfer queue P90 仍为 0、最大 2，因此它是 TTFT 的次要贡献项，不是当前主要积压点。
- 模拟 MTP 的 8 个 Decode DP rank 平均 accept length 为 3.56–3.60，accept rate 为 51.2%–52.1%，与目标 `3.61` 接近，没有看到接受率异常导致的 Decode 退化。

P/D 比例与优化建议：

- 若优先降低 TTFT，下一档 A/B 应把 P:D GPU 比例从 4:8 提高到 8:8，可比较 `1P(TP8+DPA):1D(TP8+DPA)`；另一候选是两个 P4 Prefill 服务一个 D8 Decode。两种方案都需要额外节点或重新切分 GPU，不能混入本轮固定拓扑 sweep。
- P4→P8 同时增加 Prefill attention-DP 和 P→D 并行 transfer lane 数，理论上可同时缓解 Prefill queue 与每个 Prefill rank 的 transfer 压力；是否优于 2×P4 需独立 A/B。
- 若目标是总吞吐而不是 TTFT，当前 C16 仍接近随并发翻倍，暂不应在本轮中途改比例；继续看 C32/C48/C64 的吞吐增益和 queue 拐点。
- KV transfer 优化应排在增加 Prefill 容量之后：只有当 Prefill queue 降低而 transfer latency/queue 仍高时，再独立比较 NIC mapping、lane 数或 transfer 参数，并回退无收益开关。

## AgentX C32

- 配置：`CONC=32`
- Prefill/Decode max running：32 / 32
- Prefill/Decode CUDA graph max batch：32 / 32
- 独立 launch 耗时：845.074 秒
- 实际统计窗口：1,227.681 秒
- 计分请求：1,464
- warmup 请求：354
- 错误请求：4，`InvalidInferenceResultError`
- 计分错误率：4/1,468 = 0.272%，低于 10% 门槛
- 平均 QPS：1.19234
- 总吞吐：141,649.61 tok/s
- 每 GPU 总吞吐：11,804.13 tok/s
- 每 GPU 输入吞吐：11,714.50 tok/s
- 每 GPU 输出吞吐：89.64 tok/s
- P50 TTFT：3.57753 秒
- P90 TTFT：11.71215 秒
- P95 TTFT：19.43004 秒
- P50 E2E：7.56470 秒
- P90 E2E：34.71747 秒
- P95 E2E：54.63584 秒
- P50 ITL：9.98 毫秒
- P90 ITL：12.04 毫秒
- P90 interactivity：83.00 tok/s/user
- GPU cache hit rate：94.779%
- 理论 cache hit rate：96.427%
- GPU KV cache 使用率：62%
- CPU HiCache 使用率：不适用

该点完整 PASS 并在 SSH 客户端断开期间正常落盘。4 个错误请求被保留在汇总中。

### C32 瓶颈判断与下一步

结论：**C32 的主瓶颈仍是 Prefill queue，并且 KV transfer 已从次要延迟上升为共同瓶颈；Decode 计算变慢但没有形成 compute queue。**

证据：

- 合并 4 个 Prefill DP rank 的 scheduler queue histogram 后，pooled P90 位于
  10–15 秒 bucket；8 个 Decode DP rank 的 pooled P90 仍不超过 1 毫秒。
- P50/P90 TTFT 分别从 C16 的 1.480/5.105 秒升至 3.578/11.712 秒，与 Prefill
  排队拐点一致。
- Prefill 4 个 DP rank 的平均 running request 合计约 2.98；Decode 8 个 DP rank
  合计约 12.61，低于本点 max running 32。Decode 普通 queue 大部分时间为空。
- Decode transfer queue 已持续非零：各 Decode rank 平均 0.51–0.88 个请求，rank
  P90 为 1–3，最大 3–6；prealloc queue 仍为 0。
- Prefill 侧记录 1,467 次 KV transfer，按 count 加权平均 transfer latency 约
  843.63 毫秒，平均速度约 25.40 GB/s；相较 C16 的 350.88 毫秒和 60.98 GB/s
  出现显著退化。
- P90 ITL 从 9.20 增至 12.04 毫秒，interactivity 从 108.65 降至 83.00
  tok/s/user，Decode batching 的交互性代价继续增加；但 Decode queue 仍低，因此
  不是当前 TTFT 主因。
- C16→C32 总吞吐增长 123.1%，尚未出现总吞吐平台；这说明提高负载仍能换取吞吐，
  但延迟代价已显著。
- 模拟 MTP 各 Decode rank 的平均 accept length 为 3.52–3.58，accept rate
  50.9%–51.8%，没有异常坍塌。

P/D 比例与优化建议：

- 该点不应增加 Decode 比例；普通 Decode queue 低，增加 D 不能解决 10–15 秒的
  Prefill 排队。
- 优先把 P:D GPU 比例由 4:8 调到 8:8，比较 `P8+DPA : D8+DPA`；或者比较
  `2×P4+DPA : 1×D8+DPA`。前者使 P/D rank 一一对应并增加 transfer lane，后者保留
  已验证的 P4 kernel 形态。
- 若增加 Prefill 后 scheduler queue 明显下降，但 transfer queue/速度仍无改善，
  再把 KV transfer 作为独立 A/B：核对 rank-to-NIC mapping、并行 lane 和大 KV
  传输调度。无收益的网络开关必须回退。
- 若业务只追求总吞吐，当前固定拓扑仍有并发收益；若受 TTFT 或 interactivity SLO
  约束，C32 已超过适合当前 4:8 配比的低延迟区间。

## AgentX C48

- 配置：`CONC=48`，Prefill/Decode max running 与 CUDA graph max batch 均为 48
- 基线 `PREFILL_MEM_FRACTION=0.85`：无有效性能结果
- A/B `PREFILL_MEM_FRACTION=0.80`：无有效性能结果
- A/B `PREFILL_MEM_FRACTION=0.70`：无有效性能结果
- 三次均为独立 stop → HBM idle → launch；均在 warmup 中因 Prefill rank
  `HSA_STATUS_ERROR_OUT_OF_RESOURCES` 退出，未进入 1200 秒计分窗口
- 0.70 将失败位置从 30/51 推迟到 47/51，证明增加显存余量有效但不足以承载
  本数据集同批长上下文的 per-rank KV 峰值

瓶颈结论：**C48 已超过 P4+DPA Prefill 的显存容量边界，瓶颈是 Prefill
长上下文 KV/DSA forward 的峰值显存，不是 Decode 或 KV transfer 带宽。**
本点不能以部分 warmup 推导 QPS/吞吐，故不伪造性能数据。

优化方向：按 C32 的 queue/transfer 证据和 C48 的容量证据，优先增加 Prefill
rank 数，切换到后续独立的 P8+DPA:D8+DPA sweep。继续增加 Decode 比例无效；
降低静态 KV 比例只能延迟 OOM，不能解决单 rank 获得多条超长请求时的峰值。

## AgentX C64

- 无有效性能结果
- 未重复发起已知必败的 1200 秒 run：C48 在相同数据和更低并发下已连续三次触发
  Prefill rank 显存资源耗尽，C64 的并发与 primer 数只会增加同批 KV 压力
- 该点与 C48 一并记为 P4:D8 拓扑的容量上限；保留时间用于用户已要求的 P8:D8
  扫描，不引入额外调试开关

瓶颈结论：**预期仍为 Prefill 显存容量，不能归因于 Decode 或 KV transfer。**
后续以 P8:D8 的 C48/C64 实测判断增加 Prefill rank 后是否解除容量限制。

## 问题、调试和解决方案

### 1. DPA 拓扑被 GPU 数校验错误拦截

现象：原 `lib/common.sh` 固定要求 `TP × DP == selected GPUs`。P4+DPA 的正确语义是 TP4、attention-DP4、4 GPU；DPA 在 TP 内细分 attention，不增加 GPU，因此原校验会错误要求 16 GPU。

Debug：对照 `projection_sweep.py`、projection 单测和 README，确认 TP+DPA 的 `attention-DP=TP`，且文档明确说明 DPA 不增加 GPU。

解决：DPA 开启时要求 attention-DP 整除 TP，并以 `selected GPUs == TP` 校验；DPA 关闭时保留原 `selected GPUs == TP × DP` 行为。

### 2. `glm5p2_pd` 未传递 GLM-5.2 DPA/MTP IndexShare 兼容参数

现象：旧的 DPA campaign 和 Infera E2E 配置均要求 `{"index_share_for_mtp_iteration":false}`，但本目录的 `engine.sh` 没有传递 `--json-model-override-args`。

Debug：对照现有 `glm5p2_1p1d/engine.sh`、DPA campaign 及 Infera SGLang PD E2E matrix。

解决：在现有配置中加入 `JSON_MODEL_OVERRIDE_ARGS`，由 `engine.sh` 原样传给每条 worker；`agentx_env.py` 同时检查在线命令与期望值，防止配置漂移。没有增加新的启动脚本。

### 3. 非对称 P/D 被 preflight 的等 GPU 数门禁拦截

现象：首次 `preflight.sh` 在网络测试前退出，报 `preflight requires equal P/D GPU counts; got 4 and 8`。

Debug：检查 Mooncake preflight 实现，确认工具内部按两 rank 可见 GPU 数的公共最小值形成一致 variant loop；`INFERA_PREFLIGHT_KV_GPUS` 本身也是 coverage cap。

解决：现有脚本改为验证 `min(prefill GPUs, decode GPUs)` 条公共 WRITE 路径，并在非对称时明确记录覆盖数。本轮 4 条路径全部通过字节校验。

### 4. 137 上既有 etcd 占用默认 peer port

现象：首次 launch 创建本轮 etcd 后健康检查超时。容器日志：

`creating peer listener failed: listen tcp 127.0.0.1:2380: bind: address already in use`

Debug：检查 137 上既有 `glm52-pd-etcd`。它创建于 2026-09-07，监听 client 2379，同时占用默认 peer 2380；没有配套模型容器，无法从 Docker 元数据确认操作者。

解决：不停止、不修改既有容器。为本轮现有启动脚本补充 `ETCD_PEER_PORT`，使用独立 client/peer 端口 12379/12380。第二次 launch 成功。

### 5. 同名镜像 tag 在节点间漂移

现象：137/138 的 `c29bd17-b02ab81` tag 为 `869fbd...`，140/136/139 为已验证的 `8bef14...`。

Debug：逐节点执行 `docker image inspect` 比较实际 Image ID。

解决：使用现有 `build_image.sh distribute` 从 140 同步，并由 `launch.sh` 在每次启动前重新核对所有拓扑节点 Image ID。

### 6. SSH 客户端断开期间的执行连续性

现象：C32 已启动后用户侧 SSH 断开。C32 结果完整生成并 PASS，但没有会话继续触发
C48/C64，因此 sweep 暂停在 C32。

Debug：重连后核对 `agentx_conc32.json`、runner 末尾 PASS、完整 AIPerf artifacts
及 Docker 状态。137/138 未重启；C32 容器在结果落盘约三小时后被 destroy，未发现
对应的本地 `stop.sh` 日志，无法仅凭 Docker event 确认清理者。

解决：保留 C32 原结果，不重跑、不改变参数。确认本轮容器已不存在后，从 C48 的
fresh launch 继续。后续客户端断开前不承诺依赖交互会话自动触发下一点。

### 7. 节点任务切换窗口导致 C48 首次 launch 竞态失败

现象：C48 launch 在 137/138 的旧 `glm52-1c91-*` 任务释放显存后通过 idle gate，
但 Prefill 容器启动约 104 秒后收到 SIGTERM，状态为 exit 143、`oom=false`。
Decode 容器尚在加载，服务未进入可用状态。

Debug：Docker events 显示 137 在旧任务结束 40 秒后启动了新的
`glm52-259de-c2`，本轮 idle gate 检查时新任务尚未分配显存；138 也在本轮 Decode
启动 21 秒后创建了新的 `glm52-259de-c4`。因此这是节点任务交接期间“显存短暂为
0%”造成的竞态，不是 C48 graph/max-running、模型参数或 OOM 问题。

解决：终止失败的本地 launch 等待进程，并使用现有 `stop.sh` 只删除本轮
Prefill/Decode/etcd 容器；没有触碰 `glm52-259de-*`，也没有改变任何性能开关。
后续要求节点连续一个监控周期无外部任务且显存为空后再启动，避免命中任务切换间隙。

### 8. C48 warmup 中 Prefill rank 因显存资源耗尽退出

现象：节点空闲后 C48 fresh launch 成功，Prefill、Decode 和 router 初始健康。
AgentX warmup 返回 30 个请求后停在 51 个已发送、21 个 in-flight；持续 12 分钟无新返回，
但 runner 未立即报错。router `/health` 仍返回两个 active worker，Prefill `/health`
则超时，Decode GPU 已空闲。

Debug：Prefill 日志在 2026-09-12 16:23:11 明确记录
`HSA_STATUS_ERROR_OUT_OF_RESOURCES`、`Available Free mem : 18 MB` 和
`Fatal Python error: Aborted`，栈位于 Prefill DSA forward。现场 `rocm-smi`
显示三个 Prefill GPU 满载且占用 91%--97% VRAM，另一个 rank 退出后对应 GPU
仅剩 2% VRAM；Decode 八卡为 0% 利用率。因此根因是 Prefill 高并发运行期的动态显存
余量不足，不是 Decode、KV transfer 慢或普通长请求。容器 wrapper 仍存活导致 router
worker 数不能反映内部 rank 已退出。

第一次处理：终止无效 benchmark，并用现有 `stop.sh` 只清理本轮容器；完整保留
`agentx/c48/bench/runner.log`、Prefill 容器日志和本段诊断。随后 fresh launch，
只把现有 `PREFILL_MEM_FRACTION` 从基线 0.85 降至 0.80；max-running 和 CUDA graph
max batch 仍为 48，其余开关不变。

0.80 A/B 结果：服务成功启动，Prefill 空载显存从基线约 85% 降至 80%--81%；
warmup 仍在相同的 30/51 位置停止。运行峰值四卡达到 93%--98%，随后 GPU1 rank
在 2026-09-12 16:55:26 以 `HSA_STATUS_ERROR_OUT_OF_RESOURCES`、
`Available Free mem : 0 MB` 再次 fatal abort。这说明故障可复现，且 5 个百分点的
动态余量仍不足。该无效 benchmark 已停止，产物保存在
`agentx/c48/bench-mem080/`。

0.70 A/B 结果：仍只改变同一开关。warmup 越过原失败点并返回 47/51，但 GPU1
最终继续增长到 0 MB 可用，在 2026-09-12 17:33:13 以同一 HSA 错误退出。
这说明降低静态比例只能让更多长请求完成，无法消除剩余超长请求的 per-rank
KV/DSA 峰值。无效产物保存在 `agentx/c48/bench-mem070/`。

最终处理：停止并清理 0.70 栈，不再尝试更低比例或叠加其他开关。后续 launch
不传 `PREFILL_MEM_FRACTION`，因此自动回退 `config.sh` 的基线 0.85。C48/C64
记为 P4 Prefill 的容量上限，转入 P8:D8 sweep。

## 代码改动

- `config.sh`
  - 改为 P4+DPA / D8+DPA、HiCache off、137/138、C8 默认值
  - 使用隔离端口
  - 增加已知的 IndexShare 兼容参数
- `topology.tsv`
  - Prefill 改为 137，Decode 改为 138
- `lib/common.sh`
  - 修复 DPA GPU 数语义校验
  - 校验 etcd peer port 并纳入端口冲突检测
- `engine.sh`
  - 支持 `--json-model-override-args`
- `preflight.sh`
  - 支持非对称 P/D GPU 数
- `launch.sh`
  - 显式配置独立 etcd peer port
- `tools/agentx_env.py`
  - 检查在线 model override，防止 benchmark 配置漂移
- `config.sh.example`
  - 文档化可选 model override 与 etcd peer port
- `tests/test_simplified.py`
  - 更新新拓扑断言
  - 覆盖 DPA GPU 数、非对称 preflight、peer port 和 model override

未新增测试或运行脚本。所有改动均是现有路径上的配置调整或阻塞本实验的微小修复。

## 已执行的核心命令

```bash
./check_nodes.sh
./build_image.sh distribute BUILDER_NODE=crsuse2-m2m-140
./preflight.sh

./launch.sh DECODE_SIMULATE_ACC_LEN= \
  PREFILL_MAX_RUNNING=8 PREFILL_GRAPH_MAX_BS=8 \
  DECODE_MAX_RUNNING=8 DECODE_GRAPH_MAX_BS=8
./eval/smoke.sh DECODE_SIMULATE_ACC_LEN= ...
./eval/long_context.sh DECODE_SIMULATE_ACC_LEN= ...
./eval/gsm8k.sh DECODE_SIMULATE_ACC_LEN= ...
./stop.sh ...

# 每个已运行的 C 都独立执行；C48 另以 0.80/0.70 做单变量 A/B
./launch.sh DECODE_SIMULATE_ACC_LEN=3.61 \
  PREFILL_MEM_FRACTION=<0.85|0.80|0.70> \
  PREFILL_MAX_RUNNING=C PREFILL_GRAPH_MAX_BS=C \
  DECODE_MAX_RUNNING=C DECODE_GRAPH_MAX_BS=C
./agentx_bench.sh DECODE_SIMULATE_ACC_LEN=3.61 \
  PREFILL_MEM_FRACTION=<0.85|0.80|0.70> \
  PREFILL_MAX_RUNNING=C PREFILL_GRAPH_MAX_BS=C \
  DECODE_MAX_RUNNING=C DECODE_GRAPH_MAX_BS=C \
  CONC=C DURATION=1200
./stop.sh ...
```

## 产物位置

- Preflight：`preflight-asymmetric/validation.json`
- Correctness：`correctness/`
- C8：`agentx/c8/bench/agentx_conc8.json`
- C16：`agentx/c16/bench/agentx_conc16.json`
- C32：`agentx/c32/bench/agentx_conc32.json`
- C48 基线失败：`agentx/c48/bench/runner.log`
- C48 mem=0.80 失败：`agentx/c48/bench-mem080/runner.log`
- C48 mem=0.70 失败：`agentx/c48/bench-mem070/runner.log`
- 有效点汇总：`results.csv`
- Pareto 图：`pareto.png`
