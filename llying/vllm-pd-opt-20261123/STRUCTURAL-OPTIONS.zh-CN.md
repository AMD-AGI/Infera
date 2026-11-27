# 单节点结构性改动的纸面分析（11-27，集群维护期间）

结论先说：**在一个 8 卡节点内，P4D4 + prefill PP4 基本已经是最好的结构**。换 P/D 比例要么装不下 decode 的 KV，要么把 prefill 算力减半；prefill 改 TP4 或 DP4+EP4 的通信开销都比现在 PP4 的级间等待（约 6–10%）大。更大的提升要靠：让 prefill 每个 token 更便宜（kernel 效率）、decode KV 与上下文长度脱钩（HiSparse），或者多节点。

## 依据的数字

- 模型（`GLM-5.2-MXFP4/config.json`）：78 层（前 3 层 dense）、hidden 6144、256 个路由专家 + 1 个共享专家、top-8、专家 intermediate 2048、64 个注意力头、MLA kv_lora 512 + rope 64、1 个 MTP 层、词表 154,880。权重 438 GB（408 GiB）。
- decode（DP4 + EP4，显存 0.95）：每卡权重 114.7 GiB，KV 3,490,880 token。由 R + E/4 = 114.7、R + E = 408 解出：**专家 E ≈ 391 GiB、其余 R ≈ 17 GiB**（注意力、dense、embedding、lm_head 等每卡一份）。
- prefill（PP4 × TP1）：每级权重 88.6 / 100.6 / 100.6 / 112.8 GiB（PP3 多草稿层和 lm_head）。
- C152（最好配置 a49）：prefill 算力饱和（稳态各级忙 92–95%、纯计算 85–88%）；decode KV 平均约 87%（约 1,200 万 token）。

## 各方案

| 方案 | 能否成立 | 原因 |
|---|---|---|
| **P6D2**（decode EP2） | 否 | 每张 decode 卡权重 ≈ 17 + 391 / 2 ≈ 213 GiB，KV 只剩约 40 GiB（约 100 万 token / 卡，共约 200 万），而 C152 需要约 1,200 万 → 少 6–7 倍 |
| **P2D6** | 否 | prefill 只剩 2 张卡，算力减半；而 C152 的上限正是 prefill 算力 |
| **P5D3 / P3D5 / P7D1** | 否 | decode EP 必须能整除 256 个专家（1 / 2 / 4 / 8），3、5、7 不行 |
| **prefill TP4**（代替 PP4） | 不划算 | 每层 2 次 all-reduce，每次 4096 × 6144 × bf16 = 50 MB，78 层共 156 次；按 4 卡 all-reduce 有效带宽约 300 GB/s 估算约 39 ms / 步，而一步计算约 100 ms → 约 +40% 通信（PP4 现在是 6–10% 等待）；另外专家 intermediate 2048 / 4 = 512，GEMM 更碎、效率更低 |
| **prefill DP4 + EP4**（像 decode 那样） | 不划算 | 每卡 114.7 GiB 放得下；但每层 MoE 要 all-to-all：每卡每层发出约 4096 × 8 × 6144 × 0.53 B ≈ 107 MB（FP4 打包），合并回来更大，75 个 MoE 层每步约 10–30 GB / 卡，按 100–300 GB/s 估算 +10–70%，比 PP4 的等待大 |
| **多节点（如 P8D8）** | 未知 | 11-18 试过一次 P8D8，冒烟测试就 500 了，没出分；每卡吞吐能否超过单节点需要重新做，工作量大 |

## 真正还有空间的方向

1. **prefill 每个 token 更便宜**：FP4 MoE 第一段约 2.0 PFLOPs、第二段约 1.4 PFLOPs（按 4096 token 一步的 kernel 时间算），而 MI355X 的 FP4 峰值约 10 PFLOPs，还有不少空间，但这是 aiter kernel 本身的效率，不是配置能调的。
2. **HiSparse 移植到 ROCm**：decode KV 与上下文长度脱钩，解开"prefill 一提速就撞 decode KV"的第二道上限；vLLM 里只有 CUDA 后端，估 1–2 周，可以先做 1 天的可行性验证（单卡测从 CPU 调 top-2048 KV 的延迟）。
3. **当前架构剩下的调优**（BF16 / FP8 GEMM、草稿层 MoE，约 +1.5–3%）：见 HANDOFF 0000000 节。
