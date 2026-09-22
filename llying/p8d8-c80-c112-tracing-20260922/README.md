# P8D8 C80→C112 request-level tracing 实验包

状态：**仅完成脚本，禁止自动运行**

目标是在相同 fresh deployment 中按 `C80 → C112` 顺序，定位吞吐峰值之后第一个
恶化阶段，而不是再次观察 C144 的最终饱和状态。

## 采集方法

基于 SGLang 已有能力：

- `--enable-request-time-stats-logging`
- `--enable-trace`
- `--trace-modules request,mooncake`
- Async OTLP export

派生镜像只增加：

1. ReqTimeStats log 的 per-request device/host/storage cache-tier 和 routed rank；
2. `prefill_cache_lookup` trace event。

现有 SGLang timestamps 已覆盖：

- Decode prealloc entry；
- bootstrap handshake done；
- Decode allocation done / transfer queue entry；
- transfer completion / Decode waiting entry；
- Decode forward entry；
- Prefill bootstrap/wait/forward/transfer completion；
- Mooncake send/recv spans。

## 目录

- `config/`：固定 P8D8 HiCache-on tracing 配置
- `docker/`：最小派生镜像 patch
- `scripts/otlp_jsonl_collector.py`：持久化 OTLP spans
- `scripts/parse_req_time_stats.py`：解析 per-request duration logs
- `scripts/analyze_otlp_traces.py`：按 request/rank 汇总 spans
- `scripts/compare_points.py`：C80/C112 对比
- `scripts/run_point.sh`：单点 runner
- `scripts/run_c80_c112_trace.sh`：fresh launch + C80→C112 + cleanup

## 使用顺序

只有获得 136/138 独占许可后才能执行：

```bash
ROOT=/home/liyingli/bench_agentx/baseline/Infera/bench/glm5p2_pd/results/p8d8-c80-c112-tracing-20260922

bash "$ROOT/scripts/build_trace_image.sh"
bash "$ROOT/scripts/run_c80_c112_trace.sh" --confirm-exclusive
```

运行脚本具有以下保护：

- 必须显式传入 `--confirm-exclusive`
- 两节点 GPU 必须全部空闲
- 派生镜像 digest 必须一致
- trace flags 和 collector 必须通过 live assertion
- 顺序固定为 C80→C112
- 完成或异常后只清理本实验 prefix 的容器
- 等待 HiCache VRAM 真正释放后才结束

本会话按用户要求停在“脚本完成”阶段，不启动该实验。
