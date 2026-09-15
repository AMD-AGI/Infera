# GLM-5.2-MXFP4 1P1D TP4/EP4 C8 验证报告

## 结论

目标配置已经跑通：

- 模型：`GLM-5.2-MXFP4`
- 拓扑：1 Prefill + 1 Decode
- Prefill：TP4 / EP4 / DP1，GPU 0-3，HiCache 开
- Decode：TP4 / EP4 / DP1，GPU 0-3，HiCache 关
- AgentX 并发：8
- 正式性能窗口：1200 秒
- KV 传输：Mooncake，`ionic_0..7`，GID index 1
- Decode MTP：EAGLE，5 steps / 6 draft tokens / top-k 1
- 性能口径：`SGLANG_SIMULATE_ACC_LEN=3.61`
- 正式 AgentX 结果：PASS，395/395 个计分请求成功，0 个错误

## 运行环境

- Prefill：`crsuse2-m2m-136`，数据 IP `10.245.154.168`
- Decode：`crsuse2-m2m-139`，数据 IP `10.245.159.106`
- 镜像：`infera/engine-sglang:glm52-v518-c29bd17-b02ab81`
- 两节点镜像 ID：`sha256:8bef14c08ba4764698b5b9c82f41e1a3e5c0f7bc3c4a22d161cd00dbf1feb456`
- SGLang：`0.0.0.dev16838+gc29bd17d3.d20260907`
- InferenceX commit：`6d6d296c6d323540e566b005e68c6522c816ed23`
- AgentX 记录的 CPU DRAM 配额：2752 GB

在线配置采集确认两条 worker 均为 TP4/EP4、`max_running_requests=8`；Decode CUDA graph 最大 batch 为 8。Prefill 的 `enable_hierarchical_cache=true`，Decode 为 false。

## Correctness

- Smoke：PASS，包含健康检查、P/D discovery、普通对话、工具调用与并发 burst。
- 250K long context：PASS；prompt tokens 250,016，耗时 79.234 秒。
- GSM8K 全量 1,319 题：PASS。
  - strict-match exact match：96.0576%
  - flexible-extract exact match：96.2092%

Correctness 使用真实 MTP 接受路径，即启动时显式设置 `DECODE_SIMULATE_ACC_LEN=`。

## AgentX 结果

### HiCache 关闭短测

- C8，60 秒性能窗口
- 7/7 个计分请求成功，87 个 warmup 请求不计分，0 个错误
- 总吞吐：6,124.22 tok/s
- 每 GPU 总吞吐：765.53 tok/s

该点只用于确认 C8 压测链路和长请求稳定性。AgentX 场景要求至少 900 秒才适合稳态性能结论，因此不把它作为正式结果。

### HiCache 开启正式点

- C8，实际统计窗口 1,202.803 秒
- 395/395 个计分请求成功
- 87 个 warmup 请求不计分
- 错误请求：0
- 平均 QPS：0.32833
- 总吞吐：37,010.09 tok/s
- 每 GPU 总吞吐：4,626.26 tok/s
- 每 GPU 输入吞吐：4,588.87 tok/s
- 每 GPU 输出吞吐：37.40 tok/s
- P50 TTFT：0.69058 秒
- P90 TTFT：2.01873 秒
- P95 TTFT：2.58011 秒
- P50 E2E：3.14299 秒
- P90 E2E：13.57948 秒
- P50 ITL：5.59 毫秒
- P90 ITL：6.94 毫秒
- P90 interactivity：144.09 tok/s/user
- GPU cache hit rate：97.369%
- 理论 cache hit rate：97.454%
- GPU KV cache 使用率：57%
- CPU HiCache 使用率：85.858%

## 可复现命令

从 `Infera/bench/glm5p2_pd` 执行：

```bash
./check_nodes.sh
./build_image.sh distribute BUILDER_NODE=crsuse2-m2m-140
./preflight.sh OUT_DIR=results/20260910_tp4ep4_c8_d1200_136_139/preflight-known-good-image

# Correctness：关闭 HiCache，使用真实 MTP 接受路径
./launch.sh PREFILL_HICACHE=0 DECODE_SIMULATE_ACC_LEN= \
  OUT_DIR=results/20260910_tp4ep4_c8_d1200_136_139/correctness/launch-c29
./eval/smoke.sh OUT_DIR=results/20260910_tp4ep4_c8_d1200_136_139/correctness/smoke-c29
./eval/long_context.sh OUT_DIR=results/20260910_tp4ep4_c8_d1200_136_139/correctness/long-context-c29
./eval/gsm8k.sh OUT_DIR=results/20260910_tp4ep4_c8_d1200_136_139/correctness/gsm8k-c29
./stop.sh

# 性能：Prefill 开启 HiCache，使用固定接受长度
./launch.sh PREFILL_HICACHE=1 DECODE_SIMULATE_ACC_LEN=3.61 \
  OUT_DIR=results/20260910_tp4ep4_c8_d1200_136_139/agentx-on/launch
./agentx_bench.sh PREFILL_HICACHE=1 DECODE_SIMULATE_ACC_LEN=3.61 \
  CONC=8 DURATION=1200 \
  OUT_DIR=results/20260910_tp4ep4_c8_d1200_136_139/agentx-on/bench-c8-d1200
./analyze_agentx.sh \
  RESULT_DIR=results/20260910_tp4ep4_c8_d1200_136_139/agentx-on
./stop.sh
```

## 问题与处理

较新的镜像 `infera/engine-sglang:glm52-v518-402df1e-2c71811` 在真实 MTP 长时间负载下会触发多卡 `amdgpu PERMISSION_FAULTS`。该故障在更换 Decode 节点后仍能复现，排除了单节点 IOMMU 配置是主因。回退到已通过 TP4/EP4 C8 长稳测试的 `c29bd17-b02ab81` 镜像后，完整 GSM8K 和 AgentX 正式点均通过。

`stop.sh` 还修复了 Docker 自动删除容器时 “removal already in progress” 的竞态误报；当前 13 个单元测试和全部 shell 语法检查均通过。

## 结果文件

- 正式 AgentX JSON：`agentx-on/bench-c8-d1200/agentx_conc8.json`
- 汇总 CSV：`agentx-on/results.csv`
- Pareto 图：`agentx-on/pareto.png`
- 性能指标图：`agentx-on/bench-c8-d1200/metrics_plots.png`
- Workload 分布图：`agentx-on/bench-c8-d1200/workload_distribution_plots.png`
- 正式运行日志：`agentx-on/bench-c8-d1200/runner.log`
- Prefill 在线参数：`agentx-on/launch/server-info/prefill-0.json`
- Decode 在线参数：`agentx-on/launch/server-info/decode-0.json`
- GSM8K 结果：`correctness/gsm8k-c29/results_2026-09-09T20-11-58.061386.json`
- Long-context 摘要：`correctness/long-context-c29/summary.json`

正式点完成后已停止 router、Prefill、Decode 和 etcd 容器。
