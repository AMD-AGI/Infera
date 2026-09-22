# P8D8 C80→C112 request tracing 报告

状态：**未执行；仅准备实验脚本**

## 目标

区分 C80→C112 首个吞吐/TTFT/cache-hit 拐点来自：

1. Prefix reuse distance 与 cache-tier 迁移；
2. Prefill wait/compute service time；
3. Decode bootstrap / allocation wait；
4. Decode transfer input/bytes/done-signal wait；
5. Decode waiting→running admission；
6. Rank/routing skew。

## 固定实验口径

- 同一 fresh P8D8 deployment
- Prefill 138 / Decode 136
- Prefill HiCache on，Decode HiCache off
- warmup 10/lane，profiling 3600 秒
- 顺序固定 C80→C112
- 其他配置继承 C144 已验证配方

## 待填结果

运行后由 `compare_points.py` 生成：

- request-stage p50/p90/p95/p99；
- cache device/host/storage/miss per request；
- stage duration by DP rank；
- C80→C112 dwell-time delta；
- incomplete trace / timeout path；
- 与 AgentX headline 和 server metrics 的一致性。

未经实际运行，不预填结论。
