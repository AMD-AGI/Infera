# 进度与自主决定记录（2026-11-17 17:11 起，用户离开约 10 小时）

时间都是 UTC。每个主题的细节在各自目录；这里只记时间线、自主做的决定和待你确认的事。

## 待你 review 的决定

| # | 时间 | 决定 | 理由 / 可替代方案 |
|---|---|---|---|
| D1 | 17:14 | 在 294199（v3-135）里用一个 `srun --overlap` 步骤常驻队列执行器（`setsid nohup`，与登录会话脱离），所有命令经共享目录投递 | spur 每节点只允许一个进入作业的会话，且任何步骤非零退出会结束作业；执行器恒退出 0 |
| D2 | 17:25–17:47 | 实验镜像 `infera-vllm:v0340-0c5e5e89`：v0.34.0 基础 + Mooncake v0.3.14 自编（加我们的 #4540 / #4541 与一个 include 修复）+ GEM 分配器 + Infera core（不装 `[vllm]` extra） | 镜像自带的 PyPI Mooncake 0.3.14 没有目的端网卡选择与 ACK 超时；Infera 的 faae8dd4 补丁打不到 v0.3.14；`[vllm]` extra（gaie）的 grpcio 与 vLLM 冲突 |
| D3 | 17:56 | 修 Infera：`infera/engine/vllm/args.py` 兼容 v0.34 的 CLI（`make_arg_parser` 挪了位置）；新增 `infera/engine/vllm/gem_mooncake.py` | 不修 v0.34 上起不来；GEM KV 是 crusoe 驱动的绕法 |
| D4 | 18:16 | 提交第二个节点的持有作业 294784（`amd-agentx-qos`，单节点，12 h，批处理脚本内常驻执行器）给 P8D8 用 | 三个能用的 QoS 都满、排队长，提前排；开始前单节点工作照常在 v3-135 上做 |
| D5 | 19:05 | **中止 s1 计分**（prefill 只有设备缓存）并改跑 s2 | s1 在预热中就颠簸（前缀命中 5.2%、约 8 请求 / min），测出的数没有对比意义；见 `../vllm-p4d4-crusoe-20261117/` 1a 节 |
| D6 | 19:06 | s2 同时加 CPU KV 层（1 TB）和缓存感知准入（窗口 128、阈值 0） | 两者对应 sglang recipe 的 HiCache 与 SPF；为了尽快拿到可比的基线没有分开测，之后补单项 A/B |
| D7 | 19:23 | s2 的 prefill 在 66k 冷请求时 GPU 访存故障，停掉 s2，先隔离原因 | 见下文 |

## 时间线

- 17:14 节点体检通过（amdgpu 6.16.13，8 卡显存 0%，Docker 正常）；拉 v0.34.0 镜像，模型拷到本地 NVMe（82 s）。
- 17:39 / 17:53 **聚合正确性通过**：TP4 bf16 KV 无 MTP；DP4 + FP8 KV + MTP3（GSM8K 197/200，16k 32/32 + 31/32，接受长度 3.50）。
- 18:12 **P4D4 PD 门禁通过**（Infera router + vLLM PP4 prefill + DP4 decode + Mooncake RDMA 回环 + GEM KV）。
- 18:14–19:05 s1 计分：预热中颠簸，中止（D5）。
- 19:06 s2（CPU 层 + 缓存感知准入）：prefill 起栈 472 s（锁定 1 TB 主机内存），smoke 的 66k 冷请求时 prefill PP1 `HSA_STATUS_ERROR_MEMORY_FAULT`。
- 19:23–19:29 隔离：
  - `swap_blocks_batch`（offload 拷贝，ROCm 上是 `hipMemcpyBatchAsync`）在普通 / GEM 显存 × 锁页 / `hipHostRegister` 主机内存的所有组合、
    8192 个描述符分布在 64 GB 缓冲区上，都正常（`../vllm-pd-tools-20261117/scripts/swap_fault_test.py`）。
  - 聚合模式 PP4 + DSA + FP8 KV + offload 100 GB（带 / 不带 MTP）smoke 5/5、无故障。
  - 19:29 起：offload 1 TB（无 GEM）与 GEM Mooncake + offload 100 GB 分开测。
