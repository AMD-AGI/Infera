# PD / Router 实验总报告（按分位数与运行维度修订）

阅读 [REPORT.zh-CN.md](REPORT.zh-CN.md) 或支持宽表滚动的 [REPORT.html](REPORT.html)。覆盖2026-09-22至09-29的15个测量点；本次只离线整理历史证据，没有申请GPU或重跑模型。

主表包含 Total/Output token/s/GPU、TTFT p50/p90、ITL p50/p90、P miss、D本地复用、Interactivity p50/p90，以及相对明确基线的变化。用户已确认 Interactivity 指 `intvty`。其p90是ITL p90的倒数，沿用InferenceX原口径，不是速度分布的第90百分位。

后续四个分析维度均有统一实验名称的对比表：

1. P日志的新计算tokens/batch、序列数、满chunk比例；D周期日志batch、采样running总数与P有效计算速率。
2. D不可淘汰/可淘汰/resident/空闲KV、最热rank、局部压力与预分配队列。
3. P排队和服务窗口、局部积压、D等待KV与生成时间。
4. 完成数、实际输入输出长度和吞吐，防止闭环工作量变化被当成净收益。

数据文件：

- [metrics.csv](metrics.csv) / [data.json](data.json)：客户端与请求指标，52个数值字段；保留旧版平均值和其他分位数。
- [dimensions.csv](dimensions.csv) / [dimensions.json](dimensions.json)：本次从历史日志/引擎采样提取的运行维度；JSON附原始文件SHA256、统计窗口、样本数和缺口。
- [dimension-validation.json](dimension-validation.json)：180个运行维度值与既有balance结果交叉验证一致，以及所有15行分位顺序和日志rank覆盖检查。
- [quantile-recovery-validation.json](quantile-recovery-validation.json)：B3新分位数恢复算法使用B2官方导出进行校验；B3资源末尾缺失仍明确标注。

CSV的 `*_delta_pct` 表示相对变化百分比；主表命中/miss等比例变化显示百分点pp。空值是缺失或无可比基线，不代表0。B3客户端分位数可恢复完整窗口，但配对阶段、资源和日志不完整；实际padding效率和逐步同步全局GPU batch没有足够证据，报告没有用采样running数冒充。

重新生成报告（Python 3；HTML使用环境已有的 `markdown-it-py`）：

```bash
python3 llying/experiment-summary-20261008/build_report.py
```

如需重新读取共享盘原始日志和采样，先运行以下离线提取。该过程读取数GB历史数据，不操作模型或资源：

```bash
python3 llying/experiment-summary-20261008/extract_dimensions.py
python3 llying/experiment-summary-20261008/build_report.py
```

正文在 `INTRO.zh-CN.md`、`DIMENSION-ANALYSIS.zh-CN.md`；各实验详细配对分析保留在 `ANALYSIS.zh-CN.md`，报告引用其补充证据与未完成项。C1G1旧临时汇总必要字段已归档为 `c1g1-source-extract.json`，原哈希见 `extraction-provenance.json`；本次运行维度使用共享盘完整成功运行 `c1g1r-guard-completion-31688`，不使用其失败准备轮。
