# P8D8 C144 runtime RCA 报告

状态：**首轮 partial 与 replacement full run 均已完成分析**

详细报告：

- [`analysis/full_run_rca.zh-CN.md`](analysis/full_run_rca.zh-CN.md)
- [`analysis/first_run_partial_rca.zh-CN.md`](analysis/first_run_partial_rca.zh-CN.md)
- 全量结构化数据：
  `runs/c144-resume-138-136-20260922T085229Z/analysis/full-run-data.json`
- 首轮结构化数据：
  `runs/c144-138-136-20260922T071057Z/analysis/partial-run-data.json`

首轮 client 被外部任务生命周期终止，不是 benchmark 或服务故障。后续
`c144-resume-138-136-20260922T085229Z` 不是无缝续跑，而是复用同一 warm service，
重新执行 warmup10 和完整 3600 秒 profiling；两次数据不会混合统计。
Replacement full run 已正常完成，exit code 0。

## 1. 问题

在 Prefill HiCache on 的 P8D8 sweep 中，C144 相比 C112 出现吞吐、TTFT、
Effective Prefill/Decode Concurrency 和 cache hit 的明显拐点。本实验判断该拐点
主要由以下哪一项造成：

1. Prefill/Decode 资源配比；
2. DP rank 调度不均或热点；
3. 全局/每-rank `max_running_requests`；
4. CUDA graph 覆盖边界；
5. HiCache I/O、KV handoff 或 rail 热点。

## 2. 实验身份

- 首轮 Run ID：`c144-138-136-20260922T071057Z`
- replacement Run ID：`c144-resume-138-136-20260922T085229Z`
- 首轮时间：2026-09-22 07:10:57–08:50:49 UTC
- Prefill：crsuse2-m2m-138
- Decode：crsuse2-m2m-136
- Image ID：`sha256:fd7220a57b7d3b58efd875c41f7a9ef46b93469581102d96cbeb6f5451e91d35`
- Git HEAD：见各 run 的 `snapshot/git-head.txt`
- 配置偏差：仅 Prefill 节点由原实验 137 替换为 138

## 3. 数据完整性

- [x] 配置/harness 快照及 SHA256 完整
- [x] 两节点 image ID 与预期一致
- [x] sampler 在 warmup 前启动
- [ ] engine sampler 无采样缺口：约 2%–4% scrape timeout，已原位记录
- [x] node/rail sampler 无采样缺口
- [x] warmup、profiling 时间边界可定位
- [ ] profiling/drain 完整：首轮被外部终止，replacement run 用于补齐
- [x] 首轮 request records 和 server logs 已保存
- [ ] 未发生未记录的人工干预

## 4. 结果

### 4.1 Benchmark

待填：吞吐、TTFT、ITL、完成/错误数、Effective phase concurrency、cache hit。

### 4.2 Per-rank 调度

待填：

- Prefill/Decode 各 rank running、queue 的 p50/p90/max；
- rank 间 max/min、CV、持续热点时段；
- 是否有 rank 达到每-rank request-pool 上限 32。

### 4.3 CUDA graph

待填：

- `decode_cuda_graph` / `decode_none` counter delta；
- `prefill_cuda_graph` / `prefill_none` counter delta；
- graph fallback 与 queue/吞吐变化的时间相关性；
- 启动日志实际 capture batch 列表。

### 4.4 HiCache 与 KV handoff

待填：

- Host/GPU cache usage；
- cache hit 与 Prefill queue 的时间关系；
- Decode prealloc/transfer queue；
- `WaitingForInput` timeout 的 rank、时刻和持续时间。

### 4.5 Rail

待填：

- 各 ionic rail byte rate；
- retry exhausted / ACK timeout / error delta；
- rail 热点是否与相同 DP rank 的队列热点一致。

## 5. 根因判断

待数据后按“证实 / 高置信推断 / 未区分”三层填写。

## 6. 下一步

只在 C144 无法区分候选原因时补第二个点或单变量 A/B，不预先扩展 sweep。
