# CRS PP4 实验结果，2026-10-09

C48、C80、C120 均完成各 3600 秒测量并导出结果。PP4 在 C48 小幅提升吞吐并降低 TTFT，
在 C80 吞吐下降、尾延迟升高，C120 性能明显退化。按当前配方，高并发场景没有替换原环境的性能收益。
原环境指 [`../../8p4d-atom-infera/results`](../../8p4d-atom-infera/results/)，不是 DCP prefill 变体。

## 已完成结果

下列吞吐为全部 12 张 GPU 的合计。总吞吐包括请求输入 token（含缓存命中的输入）与输出 token，
不等同于实际重新计算的 prefill token/s。吞吐使用 AgentX/AIPerf 导出的完成跨度作为分母，为 3622–3630 秒；
每档发送窗口严格为 3600 秒。

| 并发 | 总吞吐 tok/s | 输出 tok/s | TTFT p50 / p95 s | 完整响应 ITL p50 ms | 成功 / 请求错误 | 收尾取消 credits |
|---|---:|---:|---:|---:|---:|---:|
| C48 | 218576.43 | 1524.08 | 1.291 / 3.710 | 13.89 | 5406 / 0 | 2 |
| C80 | 262358.31 | 2258.17 | 6.139 / 44.174 | 17.47 | 8374 / 3 | 47 |
| C120 | 107828.92 | 1095.14 | 11.756 / 478.115 | 11.73 | 3929 / 0 | 167 |

三档的 AIPerf `metadata.submission_valid=True`，TTFT/ITL 计时覆盖率均为 100%，
时长、错误率、8P+4D 并行元数据、预热 10、K3 / 2.99 均通过既定验收。
C80 的三个错误均为客户端 `ClientOSError: Can not write request body`，错误率 3/8377=0.036%；
未定位根因，检查对应时间附近的近期服务端日志尾部未见 OOM、Traceback 或 HTTP 500。
三档均触发 `grace_period_timeout=True`；宽限期后取消 credits 单独列出，不混入聚合器的 request_errors。
C120 取消 167/4096 个已发送 credits（约 4.08%），其未完成的延迟不在成功请求分位数中。
停止引擎时有访问已释放共享内存的 TypeError，发生于正式计时结束后的退出阶段。

## 与原环境比较

| 并发 | 原总吞吐 → PP4 tok/s | 变化 | 原输出吞吐 → PP4 tok/s | 变化 | 原 TTFT p50 / p95 → PP4 s |
|---|---:|---:|---:|---:|---:|
| C48 | 210342.20 → 218576.43 | +3.91% | 1465.79 → 1524.08 | +3.98% | 2.239 / 6.439 → 1.291 / 3.710 |
| C80 | 290676.44 → 262358.31 | −9.74% | 2477.26 → 2258.17 | −8.84% | 4.646 / 15.008 → 6.139 / 44.174 |
| C120 | 264321.38 → 107828.92 | −59.21% | 2410.48 → 1095.14 | −54.57% | 9.734 / 115.526 → 11.756 / 478.115 |

C48 略有吞吐优势且 TTFT 较低；C80 吞吐下降且 TTFT 尾延迟明显升高；
C120 总吞吐下降约 59%，成功请求 TTFT p95 约为原环境的 4.14 倍，并且收尾取消较多。
C80 的完整响应 ITL p50 为 17.47 ms（原 18.61 ms），没有同步恶化。
聚合服务端 GPU cache hit 在 C80 为 63.818%（原 78.224%），CPU/external hit 为 8.555%（原 0.954%）；
这些是跨引擎聚合口径，只能作为缓存行为不同的线索，不能单凭这些指标定位性能原因。

C120 GPU cache hit 为 35.925%（原 62.617%），CPU/external hit 为 5.955%（原 4.157%）。
本次 C120 预热约 40 分 51 秒，期间曾出现 prefill0 无排队、prefill1 排队约 101 个请求；
正式计时约 8 分钟时，两个 prefill 各排队约 70 个请求，decode 仅 6 个运行请求。
这些观察说明存在明显 prefill 等待，尚未确认具体根因。C120 的 ITL p50 更低（11.73 vs 17.77 ms），
但端到端吞吐和 TTFT 明显变差，不能仅用生成阶段的 ITL 判断整体收益。

这是整套 recipe 的观察，不能当作只改变 PP 维度的消融实验：

| 项目 | 原环境正式结果 | 本次 PP4 |
|---|---|---|
| Prefill | 1×TP8+DPA，会话亲和 | 2×TP1/PP4，round-robin |
| Prefill / decode token budget | 16384 / 16384 | 8192 / 2048 |
| KV block size | 64 | 16 |
| Decode memory utilization | 0.95 | 0.94，启用相同通信组复用 |
| LMCache | 8 个私有池，每池 160 GiB；最小加载 8192 token | 2 个实例，每实例 640 GiB；四个 stage lookup；最小加载 0 |
| Forced acceptance 2.99 | decode | prefill 与 decode |
| HTTP keepalive | 默认 5 s | 900 s |
| 节点 / fabric / 构建 | 原 Slurm 环境 | CRS 136/138，本次重建固定版本镜像 |

两组都是 8P+4D、MTP K3、每 lane 预热 10、每档 3600 秒；本次没有中途调参。
两组正式结果的 amdgpu 都是 6.14.14，但网络驱动、节点和镜像依赖解析时间存在差异。
原 C80 自身记录了 4 个 ClientOSError。

## 正确性和环境

- 关闭 forced acceptance 验证：completion、chat、streaming、重复提示与 6 万 token needle（首/中/尾）通过。
- GSM8K 5-shot、200 题：195/200，strict/flexible 均为 97.5%，超过既定 93.1% 门槛。
- 136 的 GPU0–7 运行 2×TP1/PP4，层数 20/20/20/18；138 的 GPU0–3 运行 TP4+DCP4。
- 模型 `/shared_nfs/models/GLM-5.2-MXFP4`；镜像 `infera-atom:pp4-baseline-202609221542-20261009`，
  ATOM `d9f0720e2f99168f6f4e5dd07d674cba5b68d610`；PP4 未加入本次 DCP prefill 变体的 fix1 补丁。
- InferenceX `918524ff94045b3f091115f1051c22a8588edf2b`，AIPerf `754356e9a39acc6cc6afb242d123bb57c3fb6f75`。
- 运行参数为 `.tmp/crs.env` 覆盖 `config.sh`；完整镜像、依赖和输入摘要在
  [`../.record/crs-image-manifest.json`](../.record/crs-image-manifest.json)。

## 原始数据与追溯

| 并发 | Run ID | 发送窗口 UTC | 预热请求数 | 导出跨度 s |
|---|---|---|---:|---:|
| C48 | crs-pp4-full-20261009-c48 | 13:22:12.957–14:22:12.958 | 531 | 3628.64479 |
| C80 | crs-pp4-full-20261009-c80 | 15:06:43.226–16:06:43.227 | 884 | 3629.60501 |
| C120 | crs-pp4-full-20261009-c120 | 17:20:30.472–18:20:30.474 | 1332 | 3621.82325 |

聚合 JSON 和逐项验收在 [`c048`](c048/)、[`c080`](c080/)、[`c120`](c120/)。
汇总见 [`summary.md`](summary.md)，可计算的原环境对比见 [`baseline-comparison.csv`](baseline-comparison.csv)。
完整原始数据归档在本套件 `.tmp/archives/`，
排除可重建的 `agentx/tmp` 缓存，保留 runtime.env、请求记录、AIPerf 导出及完整引擎日志。

| 归档 | 字节 | SHA256 |
|---|---:|---|
| crs-pp4-c48-and-validation-20261009.tar.gz | 253865970 | 296dbedd2874e15769be6f501e67463dc2dd5a7dab0f3314e51e7cef89c2f95c |
| crs-pp4-c80-20261009.tar.gz | 460597970 | 76a4390ad42f78ab7fda1bb888bee0942ea9410570588db5092b761b2257fd4f |
| crs-pp4-c120-20261009.tar.gz | 385007846 | 5d902db64c2dde549fc7a82d3c152e186628a4243bec0879cb8a0ac6c6f6a4fe |

三份归档的 tar/gzip 内容与 SHA256 均已验证。C48 归档包含正确性验证，C120 归档还包含整组测试日志与 exit 0 记录。
三档测试已完成，测试容器均已移除；显存回收检查见 [`cleanup-status.json`](cleanup-status.json)。运行过程见
[`../.record/progress.md`](../.record/progress.md)，问题详见 [`../.record/issues.md`](../.record/issues.md)。
