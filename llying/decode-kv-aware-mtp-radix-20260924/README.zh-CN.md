# GLM-5.2 PD + DPA + MTP 下开启 decode 侧 KV-aware / radix cache

日期：2026-09-24。

## 结论

decode 侧打不开 KV-aware，是因为 SGLang 在参数解析阶段禁止 `--disaggregation-decode-enable-radix-cache` 与任何投机解码同时使用。decode 开了 MTP（EAGLE）后只能用 `ChunkCache`，没有 radix 树，也就没有 KV 事件。Infera 和测试脚本各自提前绕开了这条限制，所以没有报错，只是悄悄关闭。这条限制是上游在 #19746 中加入的一刀切规则，没有给出技术原因；上游目前只有针对 DeepSeek-V4 的 MTP 支持 PR，且未合入。

方案是增加一个默认关闭的开关 `SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC=1`，只对 EAGLE/NEXTN 且 topk=1 放开。对 GLM-5.2（DSA，非 SWA/SSM）逐项检查了投机解码与 decode radix 交汇的代码，没有发现只在这个组合下才会出错的路径。

**当前状态：已在 GPU 上通过正确性验证（B 组，decode radix 开启）。**

- 部署：135 跑 prefill，138 跑 decode，各 TP4/DP4；配置与 yihou t2f 对齐；使用真实 MTP 接受率。
- 启动检查和正确性探测全部通过。
- 完整 GSM8K 两轮都通过：随机 few-shot 0.9682（strict-match），固定 few-shot 0.9757。固定 few-shot 这一轮中，95.1% 的请求在 decode 上复用了前缀。
- 详见 [analysis/GATE-RESULTS.zh-CN.md](analysis/GATE-RESULTS.zh-CN.md)。
- 尚未验证：长前缀准确率。

**性能（AgentX C40，同镜像，各跑一次）：** 详见 [analysis/PERF-AB.zh-CN.md](analysis/PERF-AB.zh-CN.md)。

- decode radix（B）比基线（A）吞吐高 3.0%。收益来自传输量减少和预分配排队变短，ITL 基本不变。
- 再开 decode HiCache（C，write_through），吞吐比 B 低 4.5%，ITL 高 6.2%。decode 每一步多出约 1.7 ms：其中约 1–1.5 ms 与 HiCache 流量无关，约 0.5–1 ms 随写入和淘汰量增长。换写策略解决不了主要部分。
- py-spy 定位结果（`analysis/evidence/diag/`）：HiCache 代码只占 decode 调度线程约 0.3% 的时间，调度线程约 89% 的时间在等 GPU。所以多出的时间在 GPU 执行上，下一步做 GPU 层面的 B/C 对比。

**P+D 都开 HiCache（C 组）也已通过正确性验证。** 参数层面只差 decode radix（已解决）和测试脚本中的一条 guard（通过 `DECODE_EXTRA_ARGS` 绕过）。

- 固定 few-shot GSM8K：0.9727。
- 从主机加载的专项测试：96/96 回答正确；复用阶段 decode 从主机加载了 162,304 token，占 prompt 的 99.8%。
- 为了触发淘汰，decode 的显存被临时调小，因此这一组结果不能用来比较性能。
- 测试中偶发的约 11 秒延迟已查明，与本任务的改动无关：prefill 第一次遇到某个 query token 档位时，要现场编译 tilelang 稀疏注意力内核。编译期间整个 prefill 都会停下来。编译缓存留在容器里，容器重启后会重新编译。见 [analysis/PREFILL-JIT-STALL.zh-CN.md](analysis/PREFILL-JIT-STALL.zh-CN.md)。
- 详见 [analysis/DECODE-HICACHE.zh-CN.md](analysis/DECODE-HICACHE.zh-CN.md)。

**停止部署：** `scripts/bench-harness/stop.sh CONFIG=<配置文件> TOPOLOGY=scripts/bench-harness/results/decrad/topology.yihou.tsv`，配置文件是 `results/decrad/` 下对应组的文件。停止后，135 需要约 100 秒才释放显存。

在当前 1P1D、DP8、`PD_DP_RANK_AFFINITY=1` 的拓扑下，decode 事件不会改变路由选择。收益来自引擎侧：decode 复用已有前缀，减少 prefill→decode 的传输量和 decode 显存中的重复前缀。prefill 的计算量不会减少。

## 目录

| 路径 | 内容 |
|---|---|
| [analysis/ROOT-CAUSE.zh-CN.md](analysis/ROOT-CAUSE.zh-CN.md) | 四层阻断链、复现结果、其他发现 |
| [analysis/SOLUTION.zh-CN.md](analysis/SOLUTION.zh-CN.md) | 补丁设计、合理性分析、风险、GPU 验证计划、使用方法 |
| [analysis/GATE-RESULTS.zh-CN.md](analysis/GATE-RESULTS.zh-CN.md) | B 组 GPU 验证结果：启动检查、探测、两轮 GSM8K |
| [analysis/DECODE-HICACHE.zh-CN.md](analysis/DECODE-HICACHE.zh-CN.md) | decode 开 HiCache 的调研、C 组测试设计与结果 |
| [analysis/PREFILL-JIT-STALL.zh-CN.md](analysis/PREFILL-JIT-STALL.zh-CN.md) | 偶发约 11 秒延迟的根因：prefill 现场编译 tilelang 内核，附修复建议 |
| [analysis/PERF-AB.zh-CN.md](analysis/PERF-AB.zh-CN.md) | A/B/C 性能对比、C 组变慢的拆解、三种 HiCache 写策略的比较、下一步 |
| [analysis/UPSTREAM-STATUS.zh-CN.md](analysis/UPSTREAM-STATUS.zh-CN.md) | 上游 SGLang 的现状、相关 PR、差距与建议路径；PR 草稿在 [patches/upstream-pr-draft.md](patches/upstream-pr-draft.md) |
| [analysis/evidence/](analysis/evidence/) | 复现、单元测试和 GPU 验证的原始输出（`gate/` 下）；性能组在 `perf/`，定位实验在 `diag/` |
| [scripts/bench-harness/](scripts/bench-harness/) | yihou 同节点 kit 的 harness 拷贝；`results/decrad/` 下是 A/B/C 三组配置 |
| [patches/](patches/) | 01 SGLang PD 钩子；02 Infera 参数与测试；03 测试脚本 `DECODE_KV_AWARE` |
| [docker/Dockerfile](docker/Dockerfile) | 在现有镜像上叠加补丁 01、02 |
| [config/decode-kv-aware.sh](config/decode-kv-aware.sh) | 启动覆盖项 |
| [scripts/run_repro.sh](scripts/run_repro.sh) | 仅用 CPU 的复现（不启动引擎，不使用 GPU，`--network none`） |

## 未提交到 git 的原始产物

以下文件体积大（每次 AgentX 运行约 1.7 GB），只保留在共享文件系统上的这个目录里：`/home/liyingli/bench_agentx/baseline/Infera/llying/decode-kv-aware-mtp-radix-20260924/`。报告中的数字可以由已提交的汇总文件和脚本复现，但重新从原始数据计算时需要这些文件。

- `analysis/evidence/**/aiperf_artifacts/`：AgentX 的逐请求记录（`profile_export.jsonl`）、服务端指标时间序列和 timeslices。后两者因 `/home` 共享卷已满，已就地压缩为 `.gz`（`server_metrics_export.json.gz` 约 55 MB，解压后约 1.4 GB）；`scripts/server_metrics_extract.py` 可以直接读 `.gz`。
- `analysis/evidence/**/server-logs/`：prefill 和 decode 的完整服务日志。
- `analysis/evidence/gate/**/samples_gsm8k_*.jsonl`：GSM8K 的逐样本输出。分数在同目录的 `results_*.json` 中，已提交。

## 复现

```bash
llying/decode-kv-aware-mtp-radix-20260924/scripts/run_repro.sh crsuse2-m2m-138
```

## 其他发现

- Infera 的 `_decode_radix_cache_unsupported_reason` 调用了 SGLang 0.5.19 中已不存在的 `ServerArgs.get_model_config()`，对 SWA/SSM 模型的提前拦截实际上已失效。对 GLM-5.2 没有影响。
- 镜像中 Router 带有 `--pd-dp-rank-affinity`，本仓库的 `rust/router/src` 中没有这部分代码（来自 yihou 的补丁）。
