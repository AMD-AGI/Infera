# decode DP2 引擎死锁：初步分析（未定位根因）

出现于 c640 第一次正式测试（`raw/pd1p1d-c640-1123T0352/`，Slurm 36901，2026-11-23）。同配置的 c768、c892、
c640 重跑都没有复现，即 4 次正式测试中出现 1 次。

## 1. 现象

| 时间（UTC） | 事件 |
|---|---|
| 04:09:43 | AgentX c640 开始；预热请求均为 `max_tokens=1`，只做 prefill 预热缓存，decode 基本空闲 |
| 约 04:48:55 | decode 两个 DP rank 的 worker 同时停止返回（此后 decode 完成数恒为 0） |
| 04:49:55 起 | 两个 EngineCore 每 60 s 报 `No available shared memory broadcast block found in 60 seconds` |
| 04:53:53 | DP0 等待 `execute_model` 结果超时，DP1 的 `execute_dummy_batch` RPC 超时；EngineCore fatal |
| 04:53:54 | vLLM Rust 前端退出：`LLM became permanently unhealthy`，decode 服务关闭 |
| 约 04:54 | 正式计分开始，649 个请求全部收到 router 的 `502 decode … unreachable`，超过 10% 失败阈值，AIPerf 中止 |

worker 层面没有任何报错或 traceback。日志：`raw/pd1p1d-c640-1123T0352/decode.log.gz`（第 1836 行起）。

## 2. 卡住时两个 rank 在做什么

decode 为 DP2×TP1，MoE 在两个 DP rank 之间做 TP，因此两个 rank 每一步都必须同步执行前向；一个 rank 没有请求时执行
空批次（`execute_dummy_batch`）陪跑。

- **DP0**：执行一个真实批次。vLLM 在 fatal 时 dump 了该步的调度内容：1 个刚完成 KV 加载的新请求
  `chatcmpl-9e17b7c8-1212b9ae`，prompt 59,541 token，`num_computed_tokens=59540`，本步调度 6 个 token（1 个真实 token +
  5 个推测解码占位），`max_tokens=1`；`num_kv_fetch_reqs_by_stage = {waiting_to_start: 6, in_progress: 1}`，KV 使用率 1.2%。
- **DP1**：执行空批次。

两边自约 04:48:55 起都没有返回，300 s 后同时超时。

## 3. 已核实为正常、排除的方向

| 怀疑点 | 结论 | 依据 |
|---|---|---|
| 新请求带 5 个推测占位 token | 正常 | `scheduler.py` 的 `pad_spec_decode`：KV 加载完成的 decode 请求补齐到 1+5 个 token，以使用 full cudagraph |
| 该请求的 KV block 表 | 正常 | 6 组：4 组滑动窗口（窗口外为 0 号空 block）、1 组完整注意力 466 个 block（对应 59,541 token）、1 组 1 个 block |
| 跳过草稿模型在两个 rank 上不一致 | 已协调 | `v1/worker/gpu/dp_utils.py`：`allow_skip_drafts` 跨 DP 求与，空批次默认不跳过 |
| 主循环一次迭代的前向次数不一致 | 对齐 | `core.py` 的 `step_with_batch_queue`：调度 0 token 时报告未执行，`run_busy_loop` 随即补跑空批次 |
| router / hyper | 无关 | 502 是 router 连不上 decode 时立即返回 |
| Mooncake KV 传输 | 无报错 | KV fetch 各阶段正常推进 |
| 显存 | 无压力 | decode KV 使用率 1–2% |

## 4. 仍然可疑的环节

"一边真实批次、一边空批次"在三轮预热中出现了数以万计次，只有这一次卡住，更像时序竞争而非固定会触发的错误：

1. 异步调度（批次队列深度 2）下 DP0 最多同时有 2 个批次在途，DP1 逐个同步执行空批次。
2. `engines_running` 每隔 `dp_sync_interval` 步才跨 DP 同步一次。
3. 本 nightly 的 DP 协调由新的 Rust 前端（vllm-rs，`coordinator_mode=External`）负责，代码路径较新。
4. 也可能是某一张卡先发生 GPU 故障，另一张在集合通信中等待；需要内核日志确认（普通用户无权读取 `dmesg`）。

## 5. 根因定位计划

1. **现场采集**（已加入流水线，`scripts/hang_watch.sh`）：decode 日志出现新的 `No available shared memory broadcast block`
   后，对 decode 的两个 EngineCore 和两个 worker 做 `py-spy dump --native`，60 s 后再 dump 一次，同时记录 GPU 状态。只采集、不干预。
2. **内核日志**：向管理员申请 n02-25 在 04:45–04:55 的 `dmesg`，查 amdgpu 页错误 / GPU 复位。
3. **复现与二分**：单独部署 1P1D，按预热的负载形态（长 prompt、`max_tokens=1`、请求稀疏、decode 频繁在空闲与忙碌之间切换）
   连续压 1–2 小时；复现后依次关闭异步调度、DSpark、decoder replay cudagraph、Rust 前端（换 Python 前端），或改用 MoE EP。
4. **查上游**：vLLM 在 9c1d45c2 之后关于 DP + 异步调度 + 推测解码死锁的修复。

vLLM 源码（镜像内 Python 部分）：`/perf_apps/liyingli/bench_agentx/vllm-src-9c1d45c2/`。
