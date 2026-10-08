# 2026-10-08 实验总报告

优先阅读 [REPORT.zh-CN.md](REPORT.zh-CN.md)；宽表可在 [REPORT.html](REPORT.html) 中滚动查看。[metrics.csv](metrics.csv) 包含所有15个测量点的48项数值字段及相对指定基线的变化；[data.json](data.json) 保留配置、对照关系、数值和来源哈希。

报告覆盖2026-09-22至09-29已保存的数据。本次只整理和核对历史证据，没有申请GPU或重新运行性能实验。容量方案未完成、B3末尾诊断不完整、跨节点及实际输出长度差异均显式标注。

重新生成（Python 3，HTML使用环境已有的 `markdown-it-py`）：

```bash
python3 llying/experiment-summary-20261008/build_report.py
```

在 `INTRO.zh-CN.md` 和 `ANALYSIS.zh-CN.md` 维护正文，生成器从历史JSON重新读取主表数值和计算变化。C1G1临时目录中的历史汇总只提取必要字段保存为 `c1g1-source-extract.json`，原文件路径/哈希见 `extraction-provenance.json`，避免报告依赖临时文件存续。

CSV的 `_delta_pct` 始终表示相对变化百分比；Markdown/HTML里命中率、miss率的变化显示为百分点（pp）。空值表示缺失、未测或无可比基线；不代表0。

本次核对：15行请求收支与16卡吞吐分母一致；对R1、扩容下R1、R4和8K的20个指标逐项核对既有独立comparison.json一致；近期6行主指标另与既有overview/metrics.csv核对。未重跑模型或router测试，本次没有修改执行代码。
