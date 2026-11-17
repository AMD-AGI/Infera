# 单节点 P4D4：Infera + vLLM v0.34.0 + Mooncake 复现 GLM-5.2 AgentX（2026-11-17，crusoe v3-135，进行中）

对标：sglang 在本集群单节点 P4D4 C104 53,351 tok/s/GPU（`dev/pd_opt/glm_5.2_agentx` 分支 `llying/pd-repro-crusoe-20261116/`）。
方案见 `../vllm-pd-plan-20261117/PLAN.zh-CN.md` 5.3 节；脚本见 `../vllm-pd-tools-20261117/`。

## 0. 当前结论

- **PD 链路跑通，门禁干净（18:12）**：Infera rust router（与 sglang recipe 相同的参数）→ vLLM prefill TP1×PP4（GPU 0-3）→
  Mooncake RDMA 同节点回环（GEM KV）→ vLLM decode DP4（GPU 4-7，MoE 平铺 TP4）。smoke 6/6、GSM8K 196/200（乱码 0、开头坏 0）、
  16k 捞针 32/32 + 长前缀 31/32、真实接受长度 3.46。
- **GEM KV 在 vLLM 上可用**：每卡 150–163 GB 的 KV 由 `infera.engine.vllm.gem_mooncake`（`get_mem_pool_context()` 返回 GEM 内存池）分配，
  Mooncake 以 dma-buf 分 1 GiB 块注册，注册后没有 sglang 未修时那种"假 OOM"。
- 计分（C104，3600 s，模拟接受长度 2.99）18:14 起，结果待补。

## 1. 部署

| 项 | prefill | decode |
|---|---|---|
| 并行 | TP1 × PP4，`VLLM_PP_LAYER_PARTITION=20,20,20,18` | `-dp 4 -tp 1`（MoE 平铺 TP4），内置 DP 均衡关，router 用 `X-Data-Parallel-Rank` 选 rank |
| 批 / 缓存 | `--max-num-batched-tokens 8192`（chunk 8192）、前缀缓存开、调度 fcfs（vLLM 没有 SPF） | `--max-num-seqs 64`、`--stream-interval 30`、前缀缓存关（与 sglang 的 decode 相同） |
| KV | FP8 e4m3；GEM 内存池；MultiConnector[Mooncake producer] | FP8 e4m3；GEM 内存池；Mooncake consumer |
| MTP | `{"method":"mtp","num_speculative_tokens":3}`（P 也要带，生成草稿层 KV） | 同左；计分时加 `rejection_sample_method: synthetic, synthetic_acceptance_length: 2.99` |
| KV 池 | 每卡 KV 缓冲 150–163 GB（权重 107–119 GiB） | 每 rank 2,764,032 token（sglang TP4 decode 关 PTPC 时 3,043,072，少 9%） |
| 起栈 | 151 s | 682 s（每 rank 读全部 shard，约 2 s / shard） |

Mooncake：镜像 `infera-vllm:v0340-0c5e5e89`（v0.3.14 + #4540 + #4541），`MC_DISABLE_HIP=1 MC_RDMA_WRITE_PREFER_DEST_NIC=1 MC_IB_TIMEOUT=18
MC_MAX_MR_SIZE=1073741824 MC_GID_INDEX=1`，每个容器一个拓扑文件（容器内 `cuda:i` → 该 GPU 的 ionic，无备选，`results/gate-p4d4-r1/config/`）。
router：`--router-backend rust --router-policy kv-aware --kv-prefill-overlap-weight 20 --kv-decode-overlap-weight 2 --session-affinity prefill
--pd-prefill-guard-release completion --decode-input-demand on`。

## 2. 待查

- **KV 流量分布**：门禁期间 8 张 ionic 各约 5.4 GB/min、ACK 超时 0、重传 0；按"写入走目的端首选网卡"的设置，预期只在 ionic_4-7 上。
  单节点回环不受影响，两节点 P8D8 前要用 `MC_LOG_RDMA_SLICE_AFFINITY=1` 确认实际选卡。
- decode KV 池比 sglang 少 9%：显存目标（vLLM 默认 0.92，sglang 0.95）与 DP 下 dense 层每 rank 一份。
- prefill 的 aiter BF16 GEMM 多个形状无调优表（日志 "not found tuned config"），sglang recipe 带了两份 GLM-5.2 调优表。
