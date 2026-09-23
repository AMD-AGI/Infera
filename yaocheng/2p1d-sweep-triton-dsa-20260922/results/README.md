# Triton DSA 2P1D 性能结果

运行编号：`2p1d-triton-dsa-20260922T124849Z`。
Prefill 使用 137/138，Decode 使用 136，共 24 张 MI355X。
两个 DSA backend 均为 `triton`，具体实验参数见 [`experiment.json`](experiment.json)。

- [TileLang 与 Triton DSA 六面板对比图](sweep_compare.png) / [SVG](sweep_compare.svg) / [性能差异说明](sweep_compare.md)。
- [对比作图数据 CSV](sweep_compare.csv) / [Triton 相对 TileLang 的逐档变化 CSV](sweep_compare_delta.csv)。
- [`run.json`](run.json)：当前状态、拓扑、镜像和测量口径。
- [`results.csv`](results.csv)：已完成且通过验证的并发档位。
- [`sweep_results.md`](sweep_results.md)：指标汇总；运行中会随已完成档位更新。
- `c080/`、`c112/`、`c144/`、`c192/`、`c256/`：各档完成后发布聚合 JSON 和 rail 记录。

目录和 CSV 表头可能在测量前就存在；以 `run.json` 的状态和 CSV 的实测行数判断进度。
每档正式测量 3600 秒，另有 10 requests/lane 的预热。
Decode 使用模拟接受长度 3.61，结果用于性能比较。

配置快照、日志和按档状态在
[`../.tmp/results/2p1d-triton-dsa-20260922T124849Z/`](../.tmp/results/2p1d-triton-dsa-20260922T124849Z/)，
完整原始数据在同一运行编号的 `../.tmp/raw/` 下，缓存保存在 `../.tmp/cache/`。
旧 TileLang 基线见 [`../../2p1d-sweep/results/`](../../2p1d-sweep/results/README.md)。

对比图中蓝色为 TileLang，橙色为 Triton DSA；两组均为 2P1D / 24 GPU。
延迟图实线为 p50、虚线为 p90，缓存图实线为 API 实测、虚线为理论值，
Decode 面板统一使用客户端 Effective Decode Concurrency 均值。
在套件根目录运行 `python3 scripts/plot_sweep_compare.py` 可重绘 PNG、SVG 和两份 CSV。
