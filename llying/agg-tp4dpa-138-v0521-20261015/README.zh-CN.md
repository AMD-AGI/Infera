# GLM-5.2-MXFP4 aggregated（不做 PD 分离）AgentX：单机 MI355X，TP4 + DP4 attention

目标：在只有 crsuse2-m2m-138 可用期间，用 infera（sglang v0.5.21 + infera rust router）复现 InferenceX
上 Dynamo + SGLang B300 aggregated 配置（4 卡一个实例，C48 约 42k tok/s/GPU，见
`../inferencex-glm52-b300-dynamo-agg-20261015/`），并在这个场景下尽量推高 tok/s/GPU。

## 配置

一个 mixed worker，138 GPU 0–3（`config/config.agg.sh`；每轮的差异在 `config/round.<ROUND>.sh`）：

| 项 | 值 | 与 B300 的差异 |
|---|---|---|
| 并行 | TP4，`--dp-size 4 --enable-dp-attention`，`--enable-dp-lm-head`，EP1 | 同 |
| MTP | EAGLE 5/1/6，模拟接受长度 3.61（match-expected，real-draft-token） | 同 |
| 调度 | chunk 32768，max-prefill-tokens 8192，max-running 128，decode graph 最大 bs 64，`--enable-prefill-delayer --prefill-delayer-max-delay-passes 60` | 同 |
| 显存 | mem-fraction-static 0.85，KV fp8_e4m3 | 同 |
| HiCache | 每 rank 270 GB 主机池；io-backend kernel + page_first | B300 为 write_back + direct + page_first_direct；ROCm 上用 3P1D prefill 已跑通的 kernel + page_first |
| tokenizer | 1 个 worker，stream-interval 1 | 同（默认值） |
| DSA | triton prefill / decode 后端 | ROCm 专有 |
| router | infera kv-aware，`--kv-overlap-weight 1.0`，`INFERA_SESSION_AFFINITY=both`（TTL 3600 s） | B300 为 Dynamo KV router（overlap 权重默认 1.0），session affinity TTL 3600 s |

镜像、模型、AgentX 客户端（agentx-harness 1.0.9，InferenceX `918524ff`）和 harness 与 3P1D 相同
（`../3p1d-tp4-tp8-v0521-20261012/`）。每 GPU 吞吐按 InferenceX 单机口径：GPU 数 = TP × PP × PCP = 4。

测量协议与 InferenceX 的差别：InferenceX 每个并发点重新部署；这里引擎常驻，每点之前 `/flush_cache` 并重启 router
（与 3P1D 调优轮相同）。

## 复现

```bash
bash scripts/launch_agg.sh ROUND=<tag>              # etcd -> worker -> router -> 冒烟
bash scripts/agg_round.sh ROUND=<tag> VARIANT=<name> CONC=<n> [DURATION=<s>]
python3 scripts/agg_report.py results/<name>/c<n>   # 客户端 + 每 DP rank 的引擎均值
```

harness 改动（mixed 角色）：`../3p1d-tp4-tp8-v0521-20261012/scripts/bench-harness/tools/topology.py`、
`tools/agentx_env.py`（拓扑全为 mixed 时生成单机口径的 runtime.env：`IS_MULTINODE=false`、`DISAGG=false`、`TP=4`）。

## 轮次

| ROUND | 内容 | 状态 |
|---|---|---|
| a1 | HiCache write_back 270 GB | 参数校验通过后停掉（按要求先不开 HiCache 调通流程） |
| a2 | 不开 HiCache，调通流程（预热每 lane 2 个请求） | 完成：C16 / 600 s 全流程通过（不计分） |
| a3 | HiCache write_through 270 GB / rank | 启动中 |

### a2 调通结果（不计分）

`results/a2-debug2/c16/summary.txt`：12,580 tok/s/GPU，ITL p50 11.1 ms，TTFT p50 2.9 s，GPU 前缀命中
96.1%（理论 96.0%），无 retract，接受长度 ≈ 3.6。启动（无 HiCache）约 7 分钟；KV 池每 rank 2,377,856 token
（105.6 GB，fp8 MLA ≈ 44 KB/token），4 rank 共约 950 万 token。

有效并发只有 5.0（C16 的 31%；3P1D C192 为 70%）：AgentX 每条 lane 两轮之间按 trace 的间隔等待，两边每 lane
都是约 40 s 发一个请求（3P1D C192：16,953 个 / 3600 s / 192 lane；这里 240 / 600 s / 16），差别只在单个请求
本身（这里约 6 s，3P1D 约 28 s）。所以服务端跟得上时吞吐 ≈ 并发 × 每 lane 速率（这里约 790 tok/s/GPU/lane），
C48 约 38k；上限来自 KV（GPU + 主机）容量和单步算力，要靠加并发找。

## 计划

1. a2：C16 / 600 s 调通全流程（flush → router → 采样 → AgentX → 发布 → 汇总）。
2. 开 HiCache（write_through）正式测 C32、C48（3600 s，预热每 lane 10 个请求）。
3. 若 write_through 下并发上不去（KV 或主机命中受限），尽快换 write_back（B300 的做法）。
4. 之后按每点的瓶颈分析决定是否加并发、调 mem-fraction / max-running / 调度参数。
