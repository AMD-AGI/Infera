# 请求配对口径（G0 结果出来之前确定）

旧4K 与 A0 的9,574个共同 trace turn，输入token差范围仅−4到+5。源码表明 `inferencex-agentx-mvp` 强制 FIRST_TURN_PREFIX cache-bust；`build_cache_bust_marker` 将 benchmark_id/recycle_pass/trajectory_index/trace_id 的 SHA256 前12位写入 `[rid:...]`。每次 benchmark_id 默认随机，标记分词长度会变化。这与共同请求的小幅输入长度差一致；不是完整输入随机漂移的证据。

比较同时输出：

1. 严格输入、实际输出长度相同的配对。
2. 输入绝对差≤8 tokens **且**相对差≤0.1%、实际输出长度相同的配对。此容差在 G0 warmup 时确定，不依据 G0 收益挑选。

两种视图都要求相同 conversation_id/turn_index/source_trace_id/outer_idx/inner_idx/kind；重复身份剔除，未关联backend剔除。只取profiling已完成请求交集，因此具有选择效应，不能以此计算整体吞吐收益。闭环请求集合和cache状态的变化仍要由完整结果、分层与回滚复测解释；相同token长度也不能证明byte-identical。

旧4K/A0 严格共同配对3,800；容差配对8,903，实际输出长度不同551、输入相对差超限120（短输入）。输出差异另行保留，不假定全部由标记导致。client seed=42，dataset revision固定；测试输入生成流程未修改。

源码文件hash见 `client-marker-source-manifest.json`。配对脚本已用歧义身份、输入差异、输出差异、已知miss差值的构造样本检验；本次进一步用真实记录验证覆盖与长度差范围。
