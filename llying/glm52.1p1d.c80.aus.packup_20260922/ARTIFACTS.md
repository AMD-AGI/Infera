# Git内容与原始产物

新增对齐轮的完整原始导出位于同一共享root下的 `results/agentx-c80-aligned/aiperf_artifacts/`。工作区只回收其汇总和小体积指标；该轮完整校验清单为 `analysis/aligned-raw-artifacts-manifest.json`。

Git提交保留配置、构建源码/补丁、harness、固定依赖版本、复现步骤、报告、聚合JSON/CSV、指标快照、启动参数和少量关键日志。历史运行的第三方checkout不会作为嵌套Git仓库提交。

大体积AIPerf原始导出和完整worker日志不加入Git。它们仍保留在共享运行目录：

```
/perf_apps/liyingli/bench_agentx/glm52-c80-20260922/results
```

control节点 `smci355-ccs-aus-n01-33` 可访问该路径；本地完整副本也仍位于本kit的results目录，只是被.gitignore排除。目录依赖本集群访问权限，不是公开下载地址；单靠Git clone即可获得运行代码和汇总，但不能自动获得历史全部原始数据。

[raw-artifacts-manifest.json](analysis/raw-artifacts-manifest.json)记录排除文件的相对路径、字节数和SHA256。需要复核时，从共享路径取回对应文件后核验。两份server_metrics_export.json各超过1 GiB；不应直接git add或force-add。

主要被排除路径：

- `results/agentx-c80/aiperf_artifacts/`
- `results/agentx-c80-hicache/aiperf_artifacts/`
- `results/launch/server-logs/`
- `results/launch-hicache/server-logs/`
- `scripts/bench-harness/.cache/`

报告中通向上述原始目录的链接在纯Git checkout中不会存在，需先取回产物。汇总JSON、raw-record accounting audit及直接HiCache快照保留在Git中。

历史脚本产生的指标图包含有效warmup；报告中的性能表使用profiling-only指标。两者口径不同，详见analysis/hicache-results.md。
