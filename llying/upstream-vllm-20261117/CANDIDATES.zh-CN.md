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
| U9 | **OffloadingConnector 是写穿（write-through）**：算完的块立即写到 CPU 层，GPU 里的块在 CPU 上也有一份，有效容量 ≈ CPU 层；sglang HiCache 是写回（write-back，只写被换出的块），有效容量 ≈ 设备 + 主机 | `../vllm-p4d4-crusoe-20261117/` s2c / s4：命中 87–89%（AIPerf 理论 95.7%，sglang 96%），prefill 受限时吞吐 ≈ 现算 /（1 − 命中） | 写回 / 排他存放选项（GPU 换出时再写 CPU），或至少可配 | vLLM | 待量化 |
| U10 | **CPU 层容量上限与失败方式**（更正）：ROCm 上 `CPUOffloadingSpec._uses_shared_region()` 为 False，CPU 层是每个 rank 自己的锁页主机内存，不走 `/dev/shm`；2 TB 时 worker 被 SIGKILL 是主机内存被锁页占满后的 OOM（2.75 TB 物理内存）。共享 `/dev/shm` 区域只在 CUDA / XPU 与 `TieringOffloadingSpec` 下使用 | s3 | 启动时按可用物理内存检查锁页总量，超出时报错而不是被 OOM 杀掉 | vLLM | 待提 |
| U11 | **`TieringOffloadingSpec` 在 PP > 1 时起不来**：区域大小来自 `worker_kv_bytes_per_block`（`offloading/config.py`：本 worker 的 KV 张量大小 / 块数），PP 各 stage 的层数与 indexer 层分布不同，各 worker 算出的共享区域大小不一致，加入方等创建方的文件长到自己的期望值，30 s 超时（行内各 worker 槽位偏移也随之不一致） | `../vllm-p4d4-crusoe-20261117/` 1e（PP4，20/20/20/18 层，1.25 TB） | 区域大小在引擎级统一计算（与 `CPUOffloadingSpec` 一致），或加入方以创建方写入的大小为准 | vLLM | 待提 |
| U12 | **gfx950 paged MQA logits（移植 sglang #47897）**：decode / MTP verify 行比 aiter 快 1.9–2.3 倍，结果等价 | `../vllm-kf-port-20261117/` 3.2 | `rocm_fp8_paged_mqa_logits` 加 gfx950 gluon 路径（逐行展开、4 行 LPT 分组），≤ 128 行，其余回落 aiter | vLLM | 移植中 |
