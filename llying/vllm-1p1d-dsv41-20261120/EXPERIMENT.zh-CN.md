# DeepSeek-V4.1-Flash MXFP4 · Infera + vLLM 1P1D · AgentX 实验报告

日期：2026-11-20 至 2026-11-23（UTC）　负责人：liyingli

## 1. 目标

在 MI355X 上用 Infera + vLLM 做 DeepSeek-V4.1-Flash MXFP4 的 PD 分离（1P1D）部署，
配方沿用 InferenceX 当前 MI355X vLLM 最好的点（`override_dp2_umbp_c384`，agg DP2 + UMBP，
c384），在 AgentX 上测正式成绩，并和 InferenceX 及同 nightly 的 agg 复现对比。

## 2. 环境

| 项 | 值 |
|---|---|
| 集群 / 节点 | AUS MI355X，`smci355-ccs-aus-n02-25`（两个点都在这台），8× MI355X，8× ionic 400G |
| 驱动 | amdgpu 6.19.14 |
| 模型 | `/perf_apps/data/models/DeepSeek-V4.1-Flash`（MXFP4） |
| 基础镜像 | `vllm/vllm-openai-rocm:nightly-9c1d45c2…@sha256:694108707f33…`（vLLM 0.34.0rc1.dev897+g9c1d45c21） |
| Infera | 分支 `llying/example/vllm_1p1d_dsv41_flash`，镜像由 `examples/vllm_1p1d_dsv41_flash/build_image.sh` 构建 |
| 压测 | InferenceX `srt_agentic.sh`（AIPerf 1.0.14），回放 `cc-traces-weka-062126`；正式 3600 s，每条 lane 10 个预热请求 |

## 3. 部署

同一节点内 1P1D：P 用 GPU 0、1（DP2×TP1），D 用 GPU 4、5（DP2×TP1）。KV 经 Mooncake
走 RDMA 回环（`MC_DISABLE_HIP=1`，GPU k 用 `ionic_k`）。P 侧挂 standalone UMBP（449 GB 主机内存，
即"B 方案"：P 的 KV 在 GEM 显存上，UMBP 服务经 GEM HIP IPC 映射）。前端是 Infera Rust router：
kv-aware、prefill 会话亲和、prefill guard 在完成时释放。

引擎参数与 InferenceX 配方一致：KV `fp8_e4m3`，block 128，DSpark MTP K=5，MXFP4 稀疏 indexer，
MoE aiter，engram CPU offload，`max-num-batched-tokens 16384`，cudagraph 1024。D 侧另加
`--no-enable-prefix-caching --prefill-schedule-interval 11`；正式计分时 D 切到 golden 接受长度 3.51
（`rejection_sample_method: synthetic`）。完整参数见各 run 目录下的 `prefill.cmd` / `decode.cmd`，
逐项说明见示例的 `README.md` "Recipe" 一节。

## 4. 流程与门禁

每次部署都依次执行，任何一步失败即不进入 AgentX：

1. **smoke**：router 发现 1P+1D；非流式与流式回答正确；一次 16k 请求期间 prefill 节点网卡确有 RDMA 流量；decode 没有等 KV 超时；两侧均为 DP2 + `fp8_e4m3`。
2. **正确性**（`engine/accuracy.sh`）：GSM8K 200 题 ≥ 0.95 且失控输出 ≤ 2，needle 16k / 64k / 128k 各 32 条须 32/32。两个点实际均为 0 错误、0 乱码、0 失控。
3. **kv-aware**（c768 起）：8 组两轮对话，第二轮换新会话 ID，要求 8 组 prefill 选择都 `cache_hits > 0`。
4. **golden**：D 重启到接受长度 3.51，确认生效后跑 AgentX 正式计分。

## 5. 各次正式测试

| | 1P1D c384 | 1P1D c768 |
|---|---|---|
| run ID | `pd1p1d-c384-1120T0606` | `pd1p1d-c768-hyper-1121T1745` |
| 时间（UTC） | 2026-11-20 06:05–07:49 | 2026-11-21 17:45–19:54 |
| Slurm 作业 / 节点 | 36739 / n02-25 | 36888 / n02-25 |
| Infera 提交 | `c8b9acdd8` | `c33de6336` |
| P 显存 | dmabuf，GMU 0.82 | GEM，GMU 0.9（GEM HIP IPC） |
| D 显存 | GEM，GMU 0.9 | GEM，GMU 0.9 |
| P→D 下发 | 并发 | 串行（PATCHES #11） |
| kv-aware 渲染 | V4 编码器（不命中） | V4.1 编码器（PATCHES #12） |
| router hyper | 1.10.1 | 1.12.0（PATCHES #13） |
| 正确性 | GSM8K 0.975，needle 全 32/32 | GSM8K 0.975，needle 全 32/32，kv-aware 8/8 |
| 预热 | 4,249 个请求 | 8,495 个请求（2,673 s） |
| 正式计分请求 | 42,171，0 错误 | 53,690，0 错误 |

AIPerf 汇总里的 "ok/total"（42,171/46,420 和 53,690/62,185）的分母含预热请求。

**并发扫描（2026-11-23）**：在 c768 的配置上，不改任何参数、只改并发，各自全新部署
（`scripts/pipeline_conc.sh`，每个点一个独立的 holder），均在 n02-25、Infera `054985cc1`（与 `c33de6336`
代码相同，只多一次 README 提交）：

| | c640（失败） | c892 | c640（重跑） | c512 | c576 |
|---|---|---|---|---|---|
| run ID | `pd1p1d-c640-1123T0352` | `pd1p1d-c892-1123T0458` | `pd1p1d-c640-1123T0714` | `pd1p1d-c512-1123T0954` | `pd1p1d-c576-1123T1148` |
| 时间（UTC） | 03:50–04:56 | 04:57–07:12 | 07:13–09:16 | 09:52–11:45 | 11:47–13:43 |
| Slurm 作业 | 36901 | 36902 | 36903 | 36904 | 36905 |
| 正确性 | GSM8K 0.965，needle 32/32，kv-aware 8/8 | GSM8K 0.965，needle 32/32，kv-aware 8/8 | GSM8K 0.980，needle 32/32，kv-aware 8/8 | GSM8K 0.975，needle 32/32，kv-aware 8/8 | GSM8K 0.970，needle 32/32，kv-aware 8/8 |
| 预热 | 7,085 个请求，返回 7,069 时 decode 卡死 | 9,876 个请求（3,038 s） | 7,085 个请求（2,226 s） | 5,677 个请求（1,751 s） | 6,383 个请求（2,018 s） |
| 正式计分 | 649 个请求全部 502，AIPerf 中止 | 46,661，0 错误 | 54,356，0 错误 | 47,766，0 错误 | 50,903，0 错误 |

c640 第一次失败的原因见 `DECODE_HANG.zh-CN.md`。c512 起流水线自动启用只采集的挂起现场抓取，c512、c576 均未触发。

## 6. 结果

| | 1P1D c384 | **1P1D c768** | agg c384（同 nightly 复现） | InferenceX c384 |
|---|---|---|---|---|
| GPU / 每卡并发 | 4 / 96 | 4 / 192 | 2 / 192 | 2 / 192 |
| 总吞吐 tok/s/GPU | 374,108 | **459,183** | 418,000 | 427,794 |
| 输出吞吐 tok/s/GPU | 2,847 | 3,475 | 3,290 | 3,375 |
| TTFT p50 / p90（s） | 1.56 / 4.60 | 16.96 / 31.75 | 1.64 / 3.84 | 1.59 / 3.69 |
| ITL p50 / p90（ms） | 18.8 / 20.9 | 30.5 / 32.8 | 51.0 / 65.1 | 49.2 / 63.3 |
| 单用户解码 p50（tok/s） | 53.3 | 32.8 | 19.6 | 20.3 |
| 服务端前缀命中 | 98.2% | 97.9% | 95.9% | 95.0%（GPU 89.1% + 外部 5.9%） |

并发扫描结果（同一配置）：

| | c512 | c576 | c640 | c768 | c892 |
|---|---|---|---|---|---|
| 总吞吐 tok/s/GPU | 413,765 | 445,346 | **471,698** | 459,183 | 400,567 |
| 输出吞吐 tok/s/GPU | 3,200 | 3,353 | 3,549 | 3,475 | 3,116 |
| TTFT p50 / p90（s） | 2.17 / 8.58 | 2.81 / 10.94 | 5.20 / 15.88 | 16.96 / 31.75 | 42.89 / 73.69 |
| ITL p50 / p90（ms） | 27.1 / 29.0 | 28.8 / 29.9 | 29.1 / 30.6 | 30.5 / 32.8 | 27.3 / 29.9 |
| 单用户解码 p50（tok/s） | 36.9 | 34.7 | 34.4 | 32.8 | 36.6 |
| 服务端前缀命中 | 98.1% | 98.1% | 98.0% | 97.9% | 97.5% |

InferenceX 数据取自公开接口（2026-11-21 曲线，同为 `nightly-9c1d45c2`），见
`reference/inferencex_dsv41flash_mi355x_vllm_agentx.json`。逐段延迟与引擎内部指标的对比见
`ANALYSIS.zh-CN.md`。

## 7. 过程中的问题与处理

所有改动及其根因、验证、可去除条件记录在示例的 `PATCHES.md`（共 13 条）。和本报告直接相关的：

| 问题 | 处理 | PATCHES |
|---|---|---|
| vLLM 带 KV connector 时拒绝 `fp8_e4m3` | Mooncake connector 可选声明 canonical KV | #8 |
| ROCm 把 KV 补齐到 4 KiB，Mooncake 两侧分组不一致 | vLLM 补丁 | #9 |
| UMBP standalone 读不了 GEM 显存（c384 只能 P 用 dmabuf 0.82） | GEM 分配器支持 HIP IPC | #10 |
| c768 预热中止：decode 为尚未 prefill 的请求预留 KV 并开始计超时 | router 串行下发 | #11 |
| kv-aware 命中恒为 0：DSv4.1 被当成 V4 渲染 | router 加 V4.1 编码器 | #12 |
| c768 预热卡在 8,489/8,495：一个流式响应缺 chunked 结束块 | hyper 1.10 回归，升级到 1.12.0 | #13 |
| golden 阶段 router 实际没被重启、UMBP 目录从未清理 | kit 的 `pkill -f` 匹配到自身，改为方括号写法并校验 | 提交 `88395e185` |
| c640 预热末尾 decode DP2 两个 worker 同时卡死，引擎退出 | 根因未定位；流水线加入只采集的现场抓取（`scripts/hang_watch.sh`） | 见 `DECODE_HANG.zh-CN.md` |

c768 的未成功尝试，按时间顺序：

1. 11-20 08:06，并发下发：预热中止（decode KV 97% 被预留、0 在跑）→ #11。
2. 11-20 09:00，串行下发：正式计分约 20 分钟后作业被抢占（10:04）。
3. 11-20 12:47，加 #12：预热末尾 1 个请求挂起 50 分钟（#13 的 bug）。手动断开该连接导致 AgentX 判定根请求预热失败、整轮中止——这是人为失误，此后加了只采集不干预的挂起监控。
4. 11-21 11:17，加 #13：AgentX 开始 14 分钟后作业被本账号取消（非流水线发起）。
5. 11-21 17:44：成功，即本报告的 c768。

## 8. 复现

```bash
cd examples/vllm_1p1d_dsv41_flash
bash build_image.sh                      # 镜像 + 自检
bash cluster/run.sh up                   # 1P1D 部署（gem wrapper；集群相关参数见 cluster/README.md）
bash cluster/run.sh smoke && bash cluster/run.sh accuracy
bash cluster/run.sh golden && CONC=768 bash cluster/run.sh agentx
```

本次实际执行的是 `scripts/pipeline_c768_v2.sh`（c768）和 `scripts/pipeline_conc.sh`（并发扫描，`CONC=<N>`），
二者都经 Slurm holder 调用 `scripts/run_1p1d_v3.sh`；时间线见 `summary.log`。

## 9. 数据

见 `README.zh-CN.md`。完整原始数据（含 600–770 MB 的引擎指标时间序列 JSON 和逐请求 jsonl）保留在
`/perf_apps/liyingli/bench_agentx/vllm-1p1d-dsv41-20261120/logs/<run ID>/`。
