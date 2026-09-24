# 待确认：R2-on 正式性能配置

状态：**待用户确认，未启动**。这是正式性能配置；此前同意的短 smoke 单独关闭 HiCache，不作为性能数据。

## 比较方式

仅运行 R2-on 候选，不重做 A。历史参考固定为：

`/perf_apps/liyingli/bench_agentx/p8d8-adaptive-31625-20260923/runs/a0-guard-decode`

该 A0 是完整、审计通过的 C80、P/D 均 4K 原策略结果。接受跨节点误差，记录节点和驱动，不因此追加基线。

## 机器与软件

| 项目 | Prefill | Decode |
|---|---|---|
| 节点 | smci355-ccs-aus-n10-29 | smci355-ccs-aus-n01-21 |
| IP | 10.235.192.140 | 10.235.192.130 |
| Allocation | 31705 | 31706 |
| GPU | 8 × MI355X，设备0–7 | 8 × MI355X，设备0–7 |
| TP / DP / EP | 8 / 8 / 1，DP attention开启 | 相同 |

两侧属于不同 allocation，启动、存活检查及清理必须分别绑定各自的作业和容器身份，不能再使用“两个节点必须在同一个 job”这一检查。

- 镜像：`infera-sglang:aus-0922-reqtrace`
- 镜像 ID：`sha256:4f125ff9096f75611fa24caad4c01c3707dd0892251680a6483b99ffaa2a88bb`
- SGLang：`0.5.19.dev20260916+ge7f7447333`
- 模型：`/perf_apps/data/models/GLM-5.2-MXFP4`，served name=`glm5.2-mxfp4`
- Router：包含候选日志过滤的源码 `6d673a90`；在同一基线镜像中 release 构建。实际二进制 SHA256 待构建后补录，运行前核验，不使用登录节点 debug 二进制。

## 引擎参数

| 参数 | Prefill | Decode |
|---|---|---|
| **实际 chunk_size** | **4096** | **4096** |
| `--chunked-prefill-size` 启动值 | 32768（÷DP8） | 32768（÷DP8） |
| `mem_fraction_static` | 0.85 | 0.85 |
| 显式 `max_total_tokens` | 不设置，由原参数自动确定 | 不设置 |
| GPU KV 容量参考／rank | 3,143,424 tokens | 3,003,264 tokens |
| KV dtype | fp8_e4m3 | fp8_e4m3 |
| `max_running_requests` | 256 | 256 |
| CUDA graph max batch 启动设置 | 256 | 256 |
| HiCache | **开启** | 关闭 |
| HiCache 配置 | ratio=1.5；write_through / kernel / page_first | 不适用 |
| Host KV 容量参考／rank | 4,715,200 tokens | 不适用 |
| Random seed | 823508857 | 19197414 |
| MTP | 关闭 | EAGLE；steps=5、topk=1、draft tokens=6 |
| 模拟接受长度 | 不适用 | 3.61 |

GPU/host 容量是历史参考值，不是另加的扩容参数；维持0.85和原ratio，自动池容量的小幅节点差异记录比较，不因此重跑A。明显偏差先核对配置和节点状态。

DSA Prefill/Decode 均为 TileLang，保留 fused QK norm/rope、AITER allreduce fusion、grouped topk=1、`index_share_for_mtp_iteration=false` 等原设置。实际图执行方式沿用基线引擎规则，不能由 max batch=256 推断 P 一定执行 CUDA graph。

KV 传输为 Mooncake；请求为 HTTP，缓存事件为 ZMQ；不强制把 D 绑定到 P 的同号 rank。

## Router 开关

```bash
INFERA_PD_PREFILL_GUARD_RELEASE=decode  # R1关闭
INFERA_R2_DECODE_DEMAND=on             # 实际按输入需求选D
INFERA_R3_CACHE_TIERS=off
INFERA_R3_HOST_WEIGHT=0
INFERA_R4_PREFILL_WORK=off
RUST_LOG=info,infera_router::routing_experiments=warn
```

原 P/D overlap 权重分别为20/2。正常已知输入的 D 选择由 R2 决定；原逻辑保留为未知输入等情况的回退。R3关闭不等于引擎HiCache关闭。

正式阶段关闭新增逐候选日志，保留原基础 pick、P生命周期及原基线trace/metrics。R2 必需的分词和记账成本计入端到端结果，不事后扣除。

## 负载与执行

- 并发：**C80**。
- 原预热：每lane额外10请求，按原数据应为884请求。
- 正式窗口：**3600秒**，完整收尾；抢占导致不完整的窗口不报成完整性能结果。
- 数据 revision：`23f152f6f0f9399a85901b89a6458def0ef16729`。
- InferenceX commit：`918524ff94045b3f091115f1051c22a8588edf2b`。
- 输出长度预算、轨迹推进、客户端依赖锁与补丁沿用历史A0；正式负载不使用smoke的32-token输出限制。
- 正式服务按已确认配置重新启动；复用同一份已校验AITER编译库，清空逻辑缓存，完成原预热。
- 完成发压和必要取证后优先停止 Prefill，尽早释放HiCache，不让它空闲等待讨论。

重点看有效 output tokens/s/GPU、TTFT、ITL、错误/取消、实际输入输出长度和 D rank 负载分布。对历史A0的差值保留节点误差，不把差值全部解释为R2净收益。

完整展开值：[PERFORMANCE-REVIEW.json](PERFORMANCE-REVIEW.json)。实际配置源：[common.sh](config/common.sh)、[performance.sh](config/performance.sh)。
