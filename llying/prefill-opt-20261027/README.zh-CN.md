# prefill 提效（在 yihou 的 llr35 配置上），10-27 起

目标：yihou 的 4+4 C96 配置（llr35，47.8–48.0k）在 138 → 140 上复现并做 PD 正确性，作为基线；在这个工作点上找 prefill 的优化项。
每项只在 4+4 C96 上测，记录收益和适用场景（用户 10-26 定的方式）。第 6、7 步（Pareto 中段按点选 K、第二版 Pareto）暂不做（用户 10-27）。

## llr35 在 PD 下长上下文严重乱码（10-27 04:10，`correct-1027T0333`，真实接受率，138 → 140）

| 检查 | llr35 | 以前验过的干净配置（async 4+8 / custom all-reduce 4+4，K5、无 KF） |
|---|---|---|
| smoke | 5/6：`long_prefix_cold` 乱码（重复片段、`</think>` 当正文），同一前缀的 `long_prefix_repeat` 正确 | 6/6 |
| lm-eval GSM8K strict / flexible | 0.9697 / 0.9712 | 0.966–0.968 |
| 200 题探测（GSM8K，并发 64） | 0.945，**乱码 10**、跑到上限 10（生成几千 token 后退化成 "0 . 0011 . …" / "- - -"） | 0.980–0.985，乱码 0 |
| 长上下文 16k × 128（无共享前缀，prompt 中位 1.39 万 token） | 捞针 **13/64**（乱码 44、上限 49）；长前缀 GSM8K **1/64**（乱码 21、上限 64） | 捞针 64/64，长前缀 GSM8K 62–63/64，乱码 0 |
| 长上下文 100k × 32（prompt 中位 8.6 万 token） | 捞针 **2/16**（乱码 14）；长前缀 GSM8K **1/16**（乱码 8、上限 16） | — |

lm-eval 只抽最后的数字，看不出来；模拟接受率计分不看内容，也看不出来。短 prompt 的 GSM8K 在生成变长、decode 上下文超过 2048（DSA 进入稀疏 top-k）
之后开始坏，长上下文从一开始就坏，指向 decode 的长上下文路径；KF 的稀疏 MLA / indexer / top-k kernel 正是在这里接管（overlay 注释：只在
K5 的 TARGET_VERIFY 行 T = 6 × bs 上验证过，llr35 是 K3）。

**二分**（`scripts/quick_correct.sh`，每个变体只从 llr35 去掉一项、都关 HiCache；smoke + 200 题探测 + 长上下文 16k × 32）：ref（只关 HiCache，
须仍乱码）→ no-kf → no-car → no-fp8draft → k5 → no-ppcomm → no-idxshare（`config/site-bis-*.env`）。用户 04:25 决定二分提前：
4+4 C96 基线（还在等显存、未起栈）已停，二分 04:25 起，138 `$L/bisect/`（`waiter-*.out`、每个变体的 `.log` / `.probe.jsonl` / `.longctx-*.jsonl`）；
基线之后再跑（`config/plan-llr35-base.txt`）。

| 变体（llr35 去掉一项，都关 HiCache） | smoke 长前缀现算 | 200 题探测 正确 / 乱码 / 上限 | 长上下文 16k × 32：捞针 / 长前缀 GSM8K |
|---|---|---|---|
| ref（只关 HiCache） | 乱码 | 0.945 / 7 / 10 | 2/16 / 1/16 |
| no-kf（去掉 KF kernel 与 KF top-k） | 乱码 | 0.945 / 7 / 10 | 2/16 / 1/16 |
| no-car（decode 关 custom all-reduce） | 乱码 | 0.960 / 1 / 6 | 3/16 / 0/16 |

| no-fp8draft（draft MoE 不转 FP8） | 乱码 | 0.950 / 4 / 9 | 2/16 / 0/16 |
| k5（两侧 EAGLE 5/1/6） | 乱码；smoke 3/6，短 chat 也是垃圾（"Paris!!!!!riba!!!!!"） | 0.730 / 17 / 49 | 3/16 / 2/16 |
| no-idxshare（两侧 index_share false） | 乱码 | 0.945 / 3 / 7 | 3/16 / 0/16 |
| **no-ppcomm（prefill 关 `SGLANG_PP_COMM_OVERLAP`）** | **正确，smoke 6/6** | **0.975 / 0 / 3** | 2/16（乱码 2、上限 11）/ 3/16（乱码 1、上限 13） |

- **主因是 prefill 的 `SGLANG_PP_COMM_OVERLAP=1`**（yihou llr21，+1.7%）：关掉后 smoke 和短 prompt 探测恢复到干净配置的水平（0.975、0 乱码）。
  与"现算错、缓存对"一致：PP 级间通信与计算重叠，KV（含 indexer key）在算完之前被发给 decode；上下文超过 2048、DSA 开始稀疏选 token 时坏位置被选中。
- **长上下文还有第二个问题**：关掉 PP 通信重叠后乱码少了，但跑到上限的输出仍退化成 "1.1.1.11.1…" / "44489.\" 44489.\" …"，正确率很低。
- 第一轮里 KF、FP8 draft "去掉后无变化"是在 PP 通信重叠还开着时测的，被主因掩盖，对第二个问题不算排除；custom all-reduce 关掉后短 prompt 乱码 7 → 1。
- **第二轮**（`waiter3-*.out`，基础 = llr35 去掉 PP 通信重叠）：clean-k5（kit 默认配方，= 以前验过的 4+4，验证镜像与 kit 干净）、clean-k3（只换 K3）、
  再分别去掉 custom all-reduce / KF / GEMM 调优 CSV / index_share / async（`config/site-bis2-*.env`）。07:32 追加 `waiter4-*.out`：
  去掉 FP8 draft、去掉 decode 的 mem 0.95 / stream-interval 30 / CLIP 8（回到 kit 的 0.92）。

| 变体（第二轮） | smoke | 200 题探测 正确 / 乱码 / 上限 | 长上下文 16k × 32：捞针 / 长前缀 GSM8K |
|---|---|---|---|
| clean-k5（kit 默认，无 yihou 任何选项） | 6/6 | 0.990 / 0 | 16/16 / 16/16 |
| clean-k3（只换 K3） | 6/6 | 0.985 / 0 | 16/16 / 15/16 |
| no-car | 6/6 | 0.970 / 1 / 5 | 3/16 / 2/16 |
| no-kf | 6/6 | 0.935 / 6 / 12 | 4/16 / 1/16 |
| no-gemmcsv | 6/6 | 0.985 / 0 / 3 | 2/16 / 2/16 |
| no-idxshare | 6/6 | 0.935 / 6 / 7 | 6/16 / 2/16 |
| no-async / no-fp8draft / no-decodeknobs | 在跑 | | |

  - 镜像、kit、K3 本身都干净；custom all-reduce、KF、GEMM CSV、index_share 都不是第二个问题（单独去掉仍坏）。
    这个基础上短 prompt 探测的乱码在 0–6 之间波动，说明短 prompt 也没完全好。若剩下的单项都不能单独修好，改从 clean-k3 逐项加回。

## 结论（10-27 04:00，第 0 步分析；第 1 步在跑）

1. **prefill 多算的新 token**（`scripts/prefill_miss.py`，逐请求）：1P C112 的未命中 3.64% = 对话增长 2.22% + 会话在本次运行首次出现 1.17%
   + 已发过的前缀没命中 0.24%。最后一项几乎都来自每小时约 62 个请求整段前缀被驱逐（合计 0.22%，约占 prefill 新算量的 6%）；
   64 token 页对齐每请求只丢约 30 token。→ **候选：HiCache 主机层每级 250 → 500 GB**（4+4 下 138 只跑一个 prefill，内存 2.75 TB 够）。
   2P1D 的跨实例丢失更多（C176 已发前缀没命中 0.32–0.41%，首次出现的请求 68–70% 未命中，1P 是 56%）。
2. **10-24 报告里"PP0 每步多 115 ms 输入准备"要修正**（`scripts/prof_memcpy_callers.py`）：PP0 的 1.94 s HtoD 里 1,938 ms 是 15 次异步拷贝
   （HiCache `move_indices` 和 `resolve_forward_inputs`，走侧 stream，数据很小，CPU 端每次 < 0.1 ms，不阻塞调度线程），按 kernel 时间求和夸大了。
   PP0 独有的真实 GPU 工作约 30 ms / 步：`_prefix_kv_indices`（调度时拷缓存前缀的 KV 索引，约 13 ms）和 DSA indexer 取 index K 时的 Fill（约 18 ms）。
3. **开 async 后 prefill 基本是 GPU 受限**：四级时间差拉平到约 396 ms、节拍 414 ms，各级每步 GPU 忙碌 386–393 ms。剩下的大头是 GPU 上的工作：
   DSA 稀疏 attention（triton，每级每步约 100 ms）、indexer、GEMM、级间等待。
4. **KF overlay 不覆盖 prefill**：只处理 decode 的 TARGET_VERIFY 行，prefill 尺寸的调用回到原实现（`kf_sparse_mla.py` 注释 "prefill-sized launch: keep today's behaviour"）。
5. **prefill 的 BF16 GEMM 走未调优的 torch 路径**：5 种 (N, K)（6144×16384、24576×6144、2624×6144、32×6144、4096×2048），但 M 随 batch 变化，
   约 650 个不同值；aiter 按精确 M 查表，要调需先给 aiter 加 M 分桶。这几种 GEMM 约占 prefill kernel 时间 7–9%，预计收益 1–3%。
6. **TP2×PP2 重新评估：吞吐上大概率不划算**。节拍已由每级 GPU 工作决定，减少级数省不掉每级的开销；TP2 还要加每层两次 all-reduce、
   indexer 在 TP rank 间重复计算，而且 MLA 的 latent KV 在 TP rank 间复制，GPU 和主机缓存能放的 token 数减半（会加重第 1 条的驱逐）。

## 步骤与状态

| 步 | 内容 | 状态 |
|---|---|---|
| 0 | kit 移植（`2p1d-kit/llr35` = yihou kit 副本，KF overlay 与 GEMM CSV 冻结在 kit 里，md5 见 `config/`）；不占卡的分析 | 完成（上面的结论） |
| 1 | `scripts/correct_and_perf.sh`：真实接受率的 smoke、GSM8K、200 题探测、长上下文 16k × 128 与 100k × 32；然后 4+4 C96 K3（2.99）性能基线 | 正确性 03:34–04:22 完成（乱码，见上）；基线按用户决定推迟到二分之后 |
| 1' | 乱码二分（`scripts/quick_correct.sh`） | 第一轮完成（主因 PP 通信重叠）；第二轮 06:20 起，预计 09:30 前后结束 |
| 2 | 原定 TP2×PP2；按第 6 条建议改为 HiCache 500 GB（等用户确认） | 待定 |

## 复现

```bash
P=/home/liyingli/bench_agentx/baseline/Infera/llying/prefill-opt-20261027
KIT=/home/liyingli/bench_agentx/baseline/2p1d-kit/llr35
L=/mnt/m2m_nobackup/liyingli_3p1d/mix-v0522/llr
# on crsuse2-m2m-138
cd $KIT && setsid nohup bash $P/scripts/correct_and_perf.sh $KIT $P/config/site-llr35-4p4.env $L $P/config/plan-llr35-base.txt \
    > $L/driver-$(date -u +%m%dT%H%M).out 2>&1 < /dev/null &
python3 $P/scripts/prefill_miss.py $RUN/agentx/aiperf_artifacts/profile_export.jsonl LABEL
python3 $P/scripts/prof_memcpy_callers.py TRACE.json.gz HtoD          # or kernel:SUBSTR
```
