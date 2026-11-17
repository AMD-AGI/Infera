# 上游候选清单（vLLM / Mooncake / aiter / Infera，2026-11-17 起，随实验更新）

每条写明：现象 / 缺口、证据（本分支的目录）、建议的改法、目标仓库、状态。只有在本分支有可复现证据后才开 issue / PR。

| # | 缺口 | 证据 | 建议改法 | 仓库 | 状态 |
|---|---|---|---|---|---|
| U1 | **MooncakeConnector 没有自定义 KV 内存池**。connector 基类有 `get_mem_pool_context()`，MooncakeStoreConnector 已用它接 NVLINK / BAREX；PD 用的 MooncakeConnector 没有。驱动对 dma-buf 注册的 KFD 显存重复记账的主机（crusoe amdgpu 6.16）需要 GEM KV | `../vllm-p4d4-crusoe-20261117/`：Infera 子类 `infera/engine/vllm/gem_mooncake.py` 跑通 P4D4 门禁，decode KV 池保持满额 | `kv_connector_extra_config.custom_mem_pool="AMD_GEM"`，分配器来自 Mooncake 的 HIP GEM 分配器（Mooncake#5584 / #5587） | vLLM（依赖 Mooncake） | 等 U2 根因 |
| U2 | **GEM KV + OffloadingConnector：第一次 offload 写出 GPU 访存故障** | `../vllm-p4d4-crusoe-20261117/` 1b 节：只有 GEM + offload 组合复现；拷贝本身在独立测试里正常 | 先定位（150 GB 级 GEM BO、`srcAccessOrder=Any`、`/dev/shm` 共享区） | vLLM / ROCm / Infera GEM 库 | 定位中 |
| U3 | **没有 SPF**；缓存感知准入只看 GPU 前缀缓存、只分"已缓存 / 未缓存" | `../vllm-p4d4-crusoe-20261117/` 1a / 1c 节：AgentX 预热洪峰下 FCFS 进入"命中低 → 计算多 → 排队长"的自持状态；sglang 靠 SPF 避开 | 按未命中 token（含 connector / CPU 层命中）排序的准入策略 | vLLM | 待 A/B |
| U4 | **Mooncake 并发下发时，prefill 侧重排不安全**：decode 到达即预留，prefill 占着算完的 KV 等 decode 拉 | 1c 节（s2b 停顿；decode 侧证据因多 API server 指标不可靠而不全） | 串行下发（Infera router `INFERA_VLLM_MOONCAKE_SERIAL=1`，已实现）；或 decode 侧按 prefill 完成顺序预留 | Infera（已做）/ vLLM | 待 s4 验证 |
| U5 | **多 API server 时 `/metrics` 是局部的**：DP4 decode 自动开 4 个 API server，每次抓取只落到一个进程，计数与 gauge 都不完整 | 1c 节更正：decode 收到 296 个请求，抓到的成功数每 rank 只有 1 | 多 API server 下聚合 Prometheus 指标，或至少在 `/metrics` 标明局部 | vLLM | 待核实是否已有开关 |
| U6 | **DP 下 decode 起栈约 11 min**：每个 DP rank 读全部 282 个 shard（约 2 s / shard），超过默认引擎就绪超时 600 s | `../vllm-agg-correctness-20261117/` 3 节 | DP rank 只读自己需要的 MoE 分片；默认超时随模型大小放宽 | vLLM | 待量化 |
| U7 | Infera 适配 v0.34：`make_arg_parser` 挪位（已修）、`[vllm]` extra 的 grpcio 冲突、KV 事件端口只留 1 个、KV 元数据块大小（16 对实际 64） | `../vllm-image-v0340-20261117/`、P4D4 README | 已修第一项；其余待 PR | Infera | 部分完成 |
| U8 | KF kernel 特性（sglang #47276 / #47278 / #47897 / #47872 与 KF MoE）在 vLLM ROCm DSA 路径上的对应实现 | 待 decode trace | 先量化 vLLM 现有 aiter 路径（asm / Gluon 稀疏 MLA、paged MQA logits、top-k）与 sglang 的差距 | vLLM / aiter | 未开始 |
