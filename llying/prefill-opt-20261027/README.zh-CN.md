# prefill 提效（在 yihou 的 llr35 配置上），10-27 起

目标：yihou 的 4+4 C96 配置（llr35，47.8–48.0k）在 138 → 140 上复现并做 PD 正确性，作为基线；在这个工作点上找 prefill 的优化项。
每项只在 4+4 C96 上测，记录收益和适用场景（用户 10-26 定的方式）。第 6、7 步（Pareto 中段按点选 K、第二版 Pareto）暂不做（用户 10-27）。

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
| 1 | `scripts/correct_and_perf.sh`：真实接受率的 smoke、GSM8K、200 题探测、长上下文 16k × 128 与 100k × 32；然后 4+4 C96 K3（2.99）性能基线 | 03:34 起，138 `$L=/mnt/m2m_nobackup/liyingli_3p1d/mix-v0522/llr` |
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
