# P8D8 C144 runtime RCA 实验包

本目录用于复现 HiCache-on P8D8 的 C144 过载点，并区分：

1. Prefill/Decode 资源或调度不均；
2. 单个 DP rank 提前达到每-rank request-pool 上限；
3. 全局 `max_running_requests=256` 限制；
4. CUDA graph 覆盖不足或 graph boundary；
5. HiCache read/promotion、KV handoff 或单 rail 热点。

本阶段只准备了配置、脚本快照和采样器，**尚未启动服务或压测**。

## 固定实验配置

- Prefill：`crsuse2-m2m-138` / `10.245.157.237`
- Decode：`crsuse2-m2m-136` / `10.245.154.168`
- 拓扑：P8D8，GPU 0–7
- 并发：C144
- Prefill HiCache on，ratio 1.5；Decode HiCache off
- `max_running_requests=256`
- Prefill/Decode graph max BS=256
- warmup 10/lane，profiling 3600 秒
- DP rank affinity on，GPU_n 对应 ionic_n
- simulated acceptance 3.61，IndexShare off

除 Prefill 节点从原实验的 137 换成 138 外，配置与
`yihou/glm52.p8d8.agentx-sweep.packup_20260920` 的 sweep-of-record 保持一致。
单独 fresh-launch C144 不继承原 sweep 中 C80/C112 的缓存状态，因此这是机制诊断
复现，不应被冒充为严格的吞吐复测。

## 目录

- `config/`：固定配置与 138→136 topology
- `scripts/`：采样和快照脚本
- `runs/<run-id>/`：实际运行数据；脚本在运行时创建
- `REPORT.zh-CN.md`：实验报告模板及最终结论

每个 run 预期包含：

```text
runs/<run-id>/
  snapshot/                  # 配置、harness 和 git/image 身份
  launch/                    # launch.sh 输出和 server logs
  bench/                     # AgentX/AIPerf 完整产物
  sampling/
    raw/                     # 前后原始 Prometheus、server_info、container inspect
    live/engine-metrics.jsonl
    live/node-runtime.jsonl
  logs/
```

## 采样内容

`sample_engine_metrics.py` 每 2 秒保存筛选后的原始 series，完整保留所有标签，
不提前对 rank 求和：

- running/waiting、prefill/decode handoff queues；
- `token_usage`、cache hit、HiCache host used/total；
- `is_cuda_graph`、`cuda_graph_passes{mode=...}`；
- per-rank generation throughput；
- Router 的 `dynamo_*` / `infera_*` 指标。

`sample_node_runtime.py` 每 5 秒保存：

- GPU use、VRAM 和 power；
- 每条 ionic rail 的标准/HW counter；
- netdev bytes/packets/error/drop；
- 主机内存和 load。

JSONL 中的采样错误会原位记录，绝不按零处理。
HCA fault/retry counter 在实验开始前已有历史非零值，分析必须使用相邻样本或
before/after 的 delta，不能把绝对值当成本次实验新增错误。

## 运行方式

完整实验（先做节点独占门禁，再 launch、live-config 校验、采样预检和 AgentX）：

```bash
ROOT=/home/liyingli/bench_agentx/baseline/Infera/bench/glm5p2_pd/results/p8d8-c144-runtime-rca-20260922
bash "$ROOT/scripts/run_c144_diagnostic.sh"
```

wrapper 在成功或失败后都保留 live service，便于现场检查。确认不再需要后，只清理
本实验固定前缀的容器：

```bash
bash "$ROOT/scripts/stop_experiment.sh"
```

采样器也可以独立运行：

```bash
ROOT=/home/liyingli/bench_agentx/baseline/Infera/bench/glm5p2_pd/results/p8d8-c144-runtime-rca-20260922
RUN="$ROOT/runs/c144-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN"

# 0 表示持续运行，直到收到 SIGINT/SIGTERM。应在 AgentX warmup 前启动。
DURATION=0 bash "$ROOT/scripts/run_sampler.sh" "$RUN"
```

若采用定时采样，必须覆盖 C144 warmup、3600 秒 profiling 和 drain。历史 C144
warmup 中出现过 1800 秒 `WaitingForInput` timeout，不能只按 profiling 时长预算。

## 根因判据

- 全局 cap：总 running 持续达到 256；
- per-rank cap/不均衡：某 rank 持续达到 32，而其他 rank 明显低于 32；
- CUDA graph：`*_none / (*_none + *_cuda_graph)` 上升并与吞吐下降同步；
- Prefill 受限：Prefill rank 满载/排队，Decode running 偏低；
- KV handoff：Decode transfer/prealloc queue 上升，但 Decode running 不升；
- HiCache：host usage/读取压力、hit 下降与 Prefill queue 同步；
- rail 热点：单个 ionic_N 的吞吐或 retry/stall 与同 rank 队列异常同步。

C144 历史 Effective Concurrency 最大值为 248，低于全局 256；因此若复现时仍出现
拥塞，可以优先区分 per-rank 热点、graph、cache 和 handoff，而不把它简单归因于
全局 cap。
