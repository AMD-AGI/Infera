# 8P8D AgentX 结果

时间均为 UTC。过程见 [../.record/progress.md](../.record/progress.md)。

## C80（未完成，前 30 分钟）

运行编号 `c80-lmcache-c80`，作业 31690，2026-09-24 05:30:54 开始测量（每 lane 预热 10 个请求）。作业在 05:55 收到抢占信号、
06:00:48 结束，服务容器运行到约 06:01:50，测量没有完成 3600 s，InferenceX 聚合 JSON 没有生成。下表由
aiperf 的 `profile_export.jsonl` 按测量开始后前 30 分钟内完成的请求计算（`../.tmp/window_metrics.py`），
8P4D 取同一窗口；该方法对 8P4D 完整 60 分钟的计算结果与其聚合 JSON 一致（17934 对 17941 tok/s/GPU，
TTFT p50 均为 20.01 s）。

| 项目 | 配置 |
|---|---|
| Prefill | `smci355-ccs-aus-n10-29`（amdgpu 6.14.14），TP8 + DPA，eager，显存比例 0.85，会话亲和，LMCache CPU 卸载每个 DP rank 160 GiB |
| Decode | `smci355-ccs-aus-n04-21`（amdgpu 6.19.14），TP8 + DCP8，显存比例 0.62，KV 约 1400 万 token |
| MTP | K3，forced acceptance 2.99 |
| 镜像 | `infera-atom:nightly_202609221542`（`sha256:b709d3d3fc3f…`），与 8P4D 相同 |

| 部署（前 30 分钟） | GPU | 完成请求 | total tok/s/GPU | output tok/s/GPU | TTFT p50 / p90 (s) | ITL p50 (ms) |
|---|---|---|---|---|---|---|
| 8P8D（decode TP8 + DCP8，0.62） | 16 | 4809 | 19056 | 161.7 | 5.45 / 13.15 | 11.45 |
| 8P4D（decode TP4 + DCP4，0.74） | 12 | 3508 | 17778 | 156.2 | 17.95 / 30.97 | 16.12 |

- 8P8D 完成的请求多 37%（GPU 多 33%），每 GPU 吞吐高 7%，TTFT p50 为 8P4D 的 30%，ITL 低 29%。
- decode 的 KV 占用在测量开始后 2 分钟为 4.8%-10.6%，同时运行 13-17 个请求、等待 0-5 个，prefix 命中率 81%-83%；KV 容量已不构成瓶颈。
- 前 30 分钟不能代表完整一小时：8P4D 的 TTFT 随会话上下文增长而上升（完整一小时的 p50 为 20.01 s，前 30 分钟为 17.95 s），正式结论需要重测完整的 3600 s。
- 2p1d 参考（24 卡，完整一小时）：total 14975 tok/s/GPU，output 115.5，TTFT p50 2.42 s / p90 6.38 s，ITL 16.02 ms，每小时完成 10641 个请求。
- 测量前的精度检查（同一配置，关闭 forced acceptance）：GSM8K 5-shot 200 题 0.985 ± 0.009，6 万 token 大海捞针 3/3。

### 与 yihou P8D8（SGLang）C80 对比

数据来自 `../../../yihou/glm52.p8d8.agentx-sweep.packup_20260920/results/c080/agentx_conc80.json`（完整 3600 s）。

| 指标 | 8P8D ATOM（前 30 分钟） | P8D8 SGLang（3600 s） | 差异 |
|---|---|---|---|
| GPU | 16（P TP8 + DPA，D TP8 + DCP8） | 16（P TP8/DP8 + DPA，D TP8/DP8 + DPA） | |
| total tok/s/GPU | 19056 | 19513 | -2.3% |
| input tok/s/GPU | 18895 | 19356 | -2.4% |
| output tok/s/GPU | 161.7 | 157.7 | +2.5% |
| ITL p50 / p90 (ms) | 11.45 / 12.51 | 14.89 / 21.28 | p50 -23%，p90 -41% |
| TTFT p50 / p90 (s) | 5.45 / 13.15 | 5.69 / 22.12 | |
| 完成请求 | 4809（30 分钟） | 9483（60 分钟） | 按每小时折算相当 |

- 条件差异：MTP 为 K3、forced acceptance 2.99，对方为 EAGLE 5 步 / 6 draft、模拟 acceptance 3.61；按 ITL × acceptance 估算每个 decode step 约 34 ms，对方约 54 ms。二级缓存为 LMCache（每 rank 160 GiB），对方为 HiCache（ratio 1.5）。引擎与镜像不同。
- 窗口差异：8P4D 的数据显示前 30 分钟的吞吐与 ITL 与完整一小时相差约 1%（17778 对 17941 tok/s/GPU，16.12 对 16.19 ms），TTFT 随时间上升，所以吞吐和 ITL 的对比可以作为初步结论，TTFT 需要完整一小时的数据。

### 瓶颈（初步）

- decode 仍有余量：测量开始后 2 分钟 KV 占用 4.8%-10.6%、等待 0-5 个；ITL p50 11.45 ms、p90 12.51 ms，分布集中。
- TTFT 主要是等待时间：按相对上一轮新增的 token 分组（`../.tmp/ttft_breakdown.py`），新增少于 16k 的请求占 92%，TTFT p50 都约 5 s（0-1k 为 5.47 s，4k-16k 为 5.00 s），与新增 token 数无关；新增 64k 以上的请求占 1.4%，TTFT p50 24.5 s，由 prefill 计算决定。测量开始后前 5 分钟 TTFT p50 为 2.2 s，之后为 5-7 s。
- 等待发生在 prefill 端的可能性最大：8P4D C80 的 prefill 日志（同一节点、同一配置）显示，各 rank 平均 prompt 吞吐 1.4k-1.9k tok/s（约为单请求实测速度的 11%），但 13%-21% 的采样点有请求排队，最多 6-13 个。prefill 同为 8 卡 DPA 的 SGLang P8D8 TTFT p50 也是 5.7 s，而 prefill 为 16 卡的 2p1d 参考为 2.4 s。
- 可能的机制（未验证）：DPA 的 8 个 rank 每层 MoE 同步前向，任一 rank 处理长 prefill 的 16384 token 分块时，其他 rank 的短请求也要等这一步；会话固定在 rank 上，排在长 prefill 后面的请求只能等待；eager 模式每步约 200 ms 的固定开销，以及 prefill 为 MTP 额外执行的一次 decode 形态前向。8P8D 的引擎日志随抢占丢失，重测时需要记录 prefill 各 rank 的步长与排队。
