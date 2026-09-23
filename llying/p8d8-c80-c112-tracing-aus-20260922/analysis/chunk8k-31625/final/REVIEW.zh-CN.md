# C80 Prefill 8K 验证复核

自动分析于 2026-09-23 11:33:20 UTC 完成。

结论：本次 8K 未观察到吞吐收益，不能据此认定为有效优化。历史 4K 与当前 8K 的 Prefill 节点不同，不能把差异全部归因于 chunk size；下一步应在当前两节点做 4K 对照。

- 输出吞吐 2568.35 → 2537.04 token/s（-1.22%），成功请求 9651 → 9584（-0.69%）。
- TTFT mean 10.654 → 11.497 秒（+7.91%），p50 5.676 → 7.230 秒（+27.39%）；p90 25.087 → 24.797 秒（-1.16%）。
- Prefill 排队均值 5.576 → 4.484 秒（-19.58%），forward envelope 均值 3.120 → 4.504 秒（+44.37%）。
- 按相同输入长度、miss tokens、host-cache 分组，保留两侧均至少 20 请求的 22 个分组，用 min(n4k,n8k) 统一加权：排队 -20.03%，forward envelope +48.48%，transfer tail +82.98%。这些时间区间包含调度/等待，不是独占 GPU 计算或纯网络传输时间。
- miss >=32768 的请求：平均 chunk 数 18.80 → 10.15；每个 chunk envelope 均值 1.018 → 2.042 秒。chunk 数减少没有转化成总体加速。两组分别有 332/297 个请求，长度分布仍可能不同。
- 9584 条正式导出请求全部完成 P/D 关联。共发送 9597 条，收尾取消 13 条，取消等待超时后强制结束阶段；历史 4K 未导出请求为 11 条。
- 两条 InvalidInferenceResultError 经逐条检查均属于 warmup：服务只返回 usage/metadata、无实际内容。Runner 的 warmup errors=0 与导出记录的检查口径不同，不能据此说整个运行完全零异常。正式导出记录未发现 error 字段。
- 平均输入 tokens -1.43%，平均输出 tokens -0.47%；GPU cache hit 94.298% → 94.319%，host cache hit 0.597% → 0.650%。仍需控制节点和闭环轨迹进度的差异。

完整自动指标表见 CHUNK8K-COMPARISON.zh-CN.md，分组计算见 reviewed-evidence.json，chunk 证据见 chunk-evidence.json 和 baseline-chunk-evidence.json。
