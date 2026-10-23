# ATOM PR #209（GLM-5.2 4P4D）为什么能到 48.2k，和我们 SGLang 前沿差在哪（10-23）

背景：[AMD-AGI/Infera#209](https://github.com/AMD-AGI/Infera/pull/209)（10-23 02:57 合入 main）把 ATOM 4P4D kit（`examples/atom_4p4d_glm5.2`）
的峰值从 InferenceX PR #4045 的 C96 42.5k 推到 C112 48.2k tok/s/GPU。本报告只做分析，没有上机：

- ATOM 的数据取自 PR 正文和 main 上的 kit README。没有原始 JSON，派生列按 3600 s 计分窗口推算。
- 我们的数据取自 `../2p1d-v0522-20261020/results/`，派生列由 `scripts/lane_metrics.py` 算出（第 6 节）。

## 0. 结论

1. **48.2k 的口径**：
   - 引擎与拓扑：ATOM nightly 加 #209 的补丁，单机 8×MI355X。prefill 为 TP1×PP4（GPU 0–3），decode 为 TP4 + DCP4（GPU 4–7），并发 C112。
   - 指标：**输入** tok/s/GPU，包含 cache 命中的输入 token；输出只有 406 tok/s/GPU。
   - 测试条件：强制 MTP 接受长度 2.99（K3）；客户端是 InferenceX CI 的 `288471b3`（agentx-harness 1.0.11），不是我们的 `918524ff`。
2. **吞吐本质上取决于每卡能扛多少路会话**：
   - 关系式：per-GPU 输入吞吐 = 每秒完成请求 × 平均输入长度 ÷ 卡数。C112 时为 3.34 req/s × 约 115K ÷ 8 ≈ 48.2k。
   - 平均输入、输出长度由数据集决定，两边都约 116K / 975。能拉开差距的只有每卡每秒完成多少请求，也就是在延迟不爆的前提下每卡能放多少路并发。
3. **#209 的作用是让并发推到 C112 而不崩**：
   - 最大的一项是 prefill 的 CPU 层从 LMCache（同步加载）换成 Mooncake Store（异步 RDMA）。按 PR 自己的数据，C96 / C112 在 LMCache 下为 30.7k / 23.3k，换成 Store 后为 46.2k / 48.2k；C56 / C72 两者持平。
   - 其次是 decode KV 池从 160K 块扩到 198.5K 块，decode 前缀命中的悬崖从约 C85 推到约 C100。
   - decode 侧补丁把每步时间缩短 6–7%，这是低并发段比 #4045 高 5–7% 的主要来源。GC 修复带来 +1.9%（C64），token id 复用带来 +0.5%。
   - PR 没有在同一台机器、同一个客户端上逐项消融，上面的分摊是作者的判断。
4. **与我们前沿相比（同为 8 卡；我们跨两台机器，ATOM 是单机）**：在 p90 interactivity 相近处，ATOM C96 是 46.2k @ 45.6，我们 4+4 C80 是 37.0k @ 44.3（总吞吐 37.3k），**ATOM 高 25%**。负载形状相同，差距在两侧：
   - **decode**：ATOM C96 每步约 59 ms（按 ITL × 2.99 估算），每张 decode 卡服务 24 路；我们 C80 每步中位 / p90 为 71 / 78 ms（日志实测），每张 decode 卡服务 20 路，而且我们的强制接受长度更高（3.61）。每张 decode 卡的输出为 775 对 623 tok/s。
   - **prefill**：ATOM 单个 PP4 在 3.14 req/s 时 TTFT p50 为 1.66 s。我们单个 PP4 在 2.55 req/s 时已有 46% 的时间在排队，上限约 3.0 req/s（主报告估计），TTFT p50 为 3.75 s，其中一部分来自 decode KV 顶满后的等预分配。
5. **已排除或已量化的因素**（分支里已有实测，不作为主因）：
   - **DCP 本身**：我们 overlay 的 DCP4 C80 只有 30.9k，比 DPA4 低 17%（同负载下每步慢约 30%）。ATOM 的优势在于它的 DCP 实现：index-K 按序列切分、decode indexer 局部 top-k、graph 内静态缓冲、晚分配。详见 `../dcp4-improve-20261023/` 第 3 节。
   - **decode 复用前缀（radix）**：C40 下 +3.0%（`../decode-kv-aware-mtp-radix-20260924/`）。
   - **晚分配**：AUS C192 实测收益约为零（`../dcp4-improve-20261023/` 第 5 节 C）。
   - **custom AR**：DCP decode 微基准里，打开它每步省 3.5–3.9 ms（`../dcp4-pd-analysis-20261022/`）。我们在 DPA 下因为负载下乱码一直关着它。按每步 71–78 ms 算约 5%，不足以解释 25% 的差距。
6. **值得查的方向**（未做，等决定）：
   - 我们 DPA4 decode 每步 71–78 ms 的构成（最忙 rank 有 17–24 路）。对照 #209 优化的 kernel 类别逐项 profile：MLA absorbed bmm、小 M 的 FP8 GEMM、通信。
   - 单个 PP4 prefill 约 3.0 req/s 上限的构成：计算、HiCache 加载、传输各占多少，再对照 Store 的异步加载。
   - DPA 下 custom AR 乱码的根因。修好后约能拿回 5%。

## 1. 48.2k 是怎么来的

```text
一路会话（lane）的一轮：
  [等首 token: TTFT] → [解码约 975 token, 每 token 一个 ITL] → [trace 里的空闲间隔] → 下一轮（复用整段历史，约 96% 命中）

每卡每秒完成请求 ≈ 每卡并发 ÷ 一轮的时长
per-GPU 输入吞吐 ≈ 每卡每秒完成请求 × 平均输入长度
ATOM C112:  (12,027 / 3600 s) ÷ 8 = 0.418 req/s/GPU  ×  约 115K  ≈ 48.2k
```

输出 token 不到总量的 1%。所以这里比的不是 prefill 算力，而是同一张卡在延迟不爆的情况下能同时扛多少路、每路一轮多快。

## 2. #209 的优化与各自贡献（PR 自测）

| 优化 | 改了什么 | 效果 | 主要作用在 |
|---|---|---|---|
| Mooncake Store CPU 层（kit 默认 `PREFILL_CPU_TIER=store`） | 2 × 768 GiB 的 CPU KV 池（每个 NUMA 节点一个 owner）；4 个 PP stage 各走自己 GPU 的网卡，异步 RDMA 加载 | C96 / C112：LMCache 下 30.7k / 23.3k（LMCache 同步加载，约占每个 PP stage 47–52% 的时间），Store 下 46.2k / 48.2k；C56 / C72 持平 | 高并发，最大的一项 |
| decode 显存设置 | mem 0.94，`--max-num-batched-tokens 2048`，`AITER_REUSE_IDENTICAL_COMM_GROUPS=1` | KV 块约 160K → 198.5K（4 个 DCP rank 共约 12.7M token），decode 前缀悬崖约 C85 → C100 | 高并发 |
| decode 侧补丁（PR 补丁表前五行，含 GC 修复） | DCP top-k 交换改用 aiter IPC all-gather；MLA absorbed bmm 按 M 调参；DCP merge 改用自研 `alltoall_vec`；27 行 GLM-5.2 FP8 GEMM 调优（M 144–240） | B40、约 101K 上下文下，每步 42.77 → 约 40 ms（−6~7%） | 低并发段比 #4045 高 5–7% 的主要来源 |
| 融合 merge kernel `alltoall_combine_quant` | 直接从对端的注册缓冲读本 rank 的 head，按 LSE 合并并量化成 FP8 | 每层 44.7 → 38.1 µs，每步再省 0.2–0.6 ms | 小 |
| ATOM API 进程 GC 停顿 | 在线服务时不再把已完成序列堆进没人读的队列；每请求的 context gauge 改为可选 | 两个 API 进程约 45% 的墙钟时间花在 gen-2 GC 上；C64 吞吐 +1.9%，TTFT 均值 2.85 → 2.04 s | 全段 |
| decode 复用 prefill 返回的 token id | 省掉 decode 侧重新渲染模板和分词（130K token 时约 0.25 s） | C64 +0.5%，TTFT p50 −0.09 s | 小 |
| Mooncake 1 QP + 16 send worker、Rust 路由器、不再抓取 ATOM 的 `/metrics` | 稳定性与开销 | 原来的 RDMA 写挂起在 10 次部署中未再出现；其余未单独量化 | 稳定性 |

基线不统一：旧默认（C96 30.1k）用的是 `agentx.sh` 的 `918524ff` 客户端，新数据用的是 CI 客户端；LMCache 的对比数据来自"其他节点、更早的镜像和 kit"。因此在 C96 / C112 上，Store 和 decode 显存这两项的贡献没有被拆开。

## 3. 为什么"能推到 C112"是关键：两个悬崖

```text
并发 C ↑
 ├─ prefill：GPU 前缀缓存装不下 → 被挤到 CPU 层，下一轮要从 CPU 加载
 │     LMCache 同步加载：PP stage 约一半时间在等 → TTFT 爆（旧设置 C96 TTFT p50 25 s）
 │     Store 异步 RDMA：加载与计算重叠 → C112 TTFT p50 仍是 3.6 s
 └─ decode：C × 平均上下文 > decode KV 池（约 12.7M token）
       → decode 前缀命中崩塌 → prefill 要重新发送整段 prompt → TTFT 上升
       → KV 池扩大把这个悬崖从约 C85 推到约 C100
```

| C | prefill 命中 GPU / CPU / 合计 % | decode 命中 % | TTFT p50 s | 输入 tok/s/GPU |
|---:|---|---:|---:|---:|
| 56 | 84.1 / 11.9 / 96.0 | 92.3 | 0.84 | 32,925 |
| 72 | 72.7 / 21.7 / 94.4 | 84.1 | 1.05 | 40,075 |
| 96 | 62.4 / 31.2 / 93.5 | 52.9 | 1.66 | 46,234 |
| 112 | 55.5 / 36.6 / 92.1 | 25.1 | 3.64 | 48,154 |

吞吐随并发上升，直到撞上悬崖为止。#209 把两个悬崖都往后推了，所以峰值从 #4045 的 C96 移到 C112，而且到 C112 时还在上升。

## 4. 与我们前沿的逐项对比

派生指标（我们的行由 `scripts/lane_metrics.py` 算出；ATOM 的行按 PR 表格以 3600 s 推算）：

| 点 | P+D 卡 | C | 每卡并发 | 每 decode 卡并发 | 每卡 req/s | 平均输入 | 平均输出 | 输入 tok/s/GPU | 每 decode 卡输出 tok/s | ITL p50 ms | TTFT p50 s | intvty p90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 我们 4+4 C64 | 4+4 | 64 | 8.0 | 16 | 0.255 | 126.5K | 991 | 32,279 | 506 | 18.9 | 1.90 | 48.6 |
| 我们 4+4 C80 | 4+4 | 80 | 10.0 | 20 | 0.319 | 115.9K | 976 | 36,998 | 623 | 20.3 | 3.75 | 44.3 |
| 我们 2P(PP4)+1D C176 | 8+8 | 176 | 11.0 | 22 | 0.291 | 130.9K | 984 | 38,106 | 573 | 21.4 | 4.00 | 43.2 |
| ATOM C56 | 4+4 | 56 | 7.0 | 14 | 0.245 | 约 134K | 约 1006 | 32,925 | 492 | 14.17 | 0.84 | 64.0 |
| ATOM C72 | 4+4 | 72 | 9.0 | 18 | 0.329 | 约 122K | 约 978 | 40,075 | 644 | 16.75 | 1.05 | 52.1 |
| ATOM C96 | 4+4 | 96 | 12.0 | 24 | 0.392 | 约 118K | 约 988 | 46,234 | 775 | 19.88 | 1.66 | 45.6 |
| ATOM C112 | 4+4 | 112 | 14.0 | 28 | 0.418 | 约 115K | 约 972 | 48,154 | 812 | 21.21 | 3.64 | 42.1 |

几点观察：

- **同一 interactivity 下的吞吐**：ATOM C72 在 interactivity 更好的情况下（52.1 对 44.3），吞吐也高于我们 C80（40.1k 对 37.0k）；ATOM C96 和我们 C80 的 interactivity 接近，吞吐高 25%。
- **decode 每步时间**：ATOM 按 ITL × 2.99 估算，C72 / C96 / C112 约为 50 / 59 / 63 ms。用同样方法估算我们 C80 得 73 ms，与日志实测的 71–78 ms 吻合，说明这个估算方法可用。

结构差异（都是配置事实，哪项是主因见第 0 节第 5、6 条）：

| | ATOM 4P4D（#209 默认） | 我们 4+4（kit `v6-pp4`，站点 `site-1p-pp4-ck8k-d4.env`） |
|---|---|---|
| 部署 | 单机 8 卡，P → D 走本机网卡 | 跨两台：prefill 在 138 的 GPU 4–7，decode 在 140 的 GPU 4–7 |
| prefill | 1 个 TP1×PP4，eager（PP 下不用 CUDA graph），`max-num-batched-tokens` 8192 | 1 个 PP4，chunk 8k，prefill 带 MTP |
| prefill CPU 层 | Mooncake Store 2 × 768 GiB，异步 RDMA | HiCache，每 rank 250 GB，write_back |
| decode | TP4 + DCP4，每个请求的 KV 按 token 切到 4 个 rank | TP4 + DPA4，每个 rank 持有自己请求的全部 KV |
| decode KV 池 | 约 12.7M token（0.94） | 约 12.0M token（`decode_kv.py` DPA4 每 rank 3,001,920） |
| decode 是否复用前缀 | 是（92% → 25%） | 否：decode 关闭 KV events、没开 HiCache，用 chunk cache |
| decode 何时分配 KV | prefill 算完后（路由器等 prefill 响应） | 握手时（`early`） |
| custom all-reduce | 开（aiter） | 关（`--disable-custom-all-reduce`，DPA 下负载时乱码） |
| 强制接受长度 | 2.99（K3） | 3.61（EAGLE 5/1/6） |
| 客户端 | InferenceX CI `288471b3`，agentx-harness 1.0.11 | `918524ff` |

## 5. 可比性注意事项

- **客户端不同**：两边平均输入（约 116K / 118K）和平均输出（约 975）接近，负载形状可比，但不是同一个客户端版本。
- **强制接受长度不同**：ATOM 是 2.99，我们是 3.61。这一项对我们的 decode 更有利，所以不能用它解释 ATOM 领先。
- **ATOM 的测试设置**：关闭了 session affinity，路由器 `--breaker-failure-threshold 0`，4 个点分布在 3 台机器上。ATOM 的派生列假设计分窗口恰好 3600 s（我们的 JSON 里约为 3630 s）。
- **#209 的分摊**：见第 2 节末尾，PR 没有逐项消融。

## 6. 复现

我们的派生行（在 `llying/` 下执行）：

```bash
R=2p1d-v0522-20261020/results
python3 atom-pr209-analysis-20261023/scripts/lane_metrics.py \
  "4+4 C64=$R/pp4-ck8k-d4-c64/agentx_conc64.json" \
  "4+4 C80=$R/pp4-ck8k-d4-c80/agentx_conc80.json" \
  "2P(PP4)+1D C176=$R/pp4-ck8k-2p-c176/agentx_conc176.json"
```

ATOM 的行：每卡 req/s = 请求数 ÷ 3600 ÷ 8；平均输入 = 输入 tok/s/GPU × 8 ÷ (请求数 ÷ 3600)；每 decode 卡输出 = 输出 tok/s/GPU × 8 ÷ 4；
每步时间 ≈ ITL p50 × 2.99。原始数字见 [PR #209](https://github.com/AMD-AGI/Infera/pull/209) 第 5 节和 main 上 `examples/atom_4p4d_glm5.2/README.md`
的 "Reference numbers"。
