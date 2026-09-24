# R2：无 HiCache 短 smoke，再直接与历史 4K 对照

2026-09-24，按用户最新安排准备。没有启动新 GPU 任务。作业31705目前 PENDING，节点与 release 二进制绑定尚未完成；调度预测不是实际分配。

## 两阶段配置

| 配置 | R2-on smoke | 正式 R2-on 4K |
|---|---|---|
| P/D 实际 chunk | 4096 / 4096 | 4096 / 4096 |
| P/D 启动 chunk 参数 | 32768 / 32768（DP=8） | 相同 |
| 引擎 P HiCache | 关闭 | 恢复历史 ratio=1.5、write_through/kernel/page_first |
| 引擎 D HiCache | 关闭 | 关闭 |
| R1 / R2 / R3 / R4 | decode / on / off / off | 相同 |
| 负载 | C80，80条有界功能请求，最多32输出tokens | C80，历史884条预热，3600秒正式窗口 |
| 新增逐候选日志 | 开启，核对真实派单和记账 | 关闭，保留原基础日志 |

smoke 以镜像、模型、编译缓存就绪后约10–15分钟内完成服务启动和短验证为目标；总预算900秒，不能把短测自动延长为完整benchmark。若准备条件不足或启动超时，停止并报告。smoke不运行884条预热或3600秒窗口。

smoke关闭HiCache只用于降低启动/清理成本，其结果不参与性能比较。正式实验重新按历史基线配置启动服务，包含P的HiCache；不开新的A，直接比较完整历史A0。R2只依赖输入长度及请求生命周期，不要求host缓存。

## 历史参考与固定项

历史参考拟使用 `p8d8-adaptive-31625-20260923/runs/a0-guard-decode`，即完整、审计通过的4K原策略A0。它已经包含P生命周期基础日志，适合作为本次直接比较的明确参考；不使用被中断的8K旁路数据。

- 镜像：`infera-sglang:aus-0922-reqtrace`，ID `sha256:4f125ff9096f75611fa24caad4c01c3707dd0892251680a6483b99ffaa2a88bb`。
- 模型：`/perf_apps/data/models/GLM-5.2-MXFP4`；每侧8张MI355X，TP8/DP8/EP1。
- P/D mem_fraction=0.85；正式阶段预期GPU容量为3,143,424 / 3,003,264 tokens/rank，P host为4,715,200。
- P/D seed=823508857 / 19197414；调度上限均256。
- DSA TileLang，KV fp8_e4m3；Decode EAGLE steps=5、topk=1、draft tokens=6，模拟接受长度3.61。
- 原P/D overlap权重20/2；请求HTTP、事件ZMQ，P/D rank不强制绑定。
- 原数据revision、InferenceX commit、客户端约束和基础trace保持一致。

用户已明确接受节点误差，要求不因换节点而反复重做基线。后续固定使用历史完整A0，只跑优化候选；节点、驱动和启动信息用于记录与解释，不作为追加A的理由。实际节点落地、binary哈希确定后，正式配置仍需先交给用户确认，再进入性能发压。不得把历史比较的全部差值自动归因于R2。

## 观测开销的区别

R2的输入长度计算、需求账本和选rank是新算法本身的成本，应计入端到端性能。R3分层目录不是R2所需，因此两阶段均关闭。逐候选诊断日志属于可关闭的观察开销，正式阶段用 `RUST_LOG=info,infera_router::routing_experiments=warn` 关闭；原 `infera_router::policy` 基础日志仍开启。

这不会让新代码与历史版本的执行开销完全相同，也不应把R2自身必要成本隐藏或事后扣除。

## 配置文件使用边界

[common.sh](config/common.sh)、[smoke.sh](config/smoke.sh)、[performance.sh](config/performance.sh) 是待绑定、待验收的配置，不会自动启动服务。必须显式提供新分配的节点、地址、独立runtime路径和已验证release二进制；没有沿用失效的31699节点绑定。

运行器还需落实900秒短测预算、无HiCache的验收条件，以及已有的用户配置批准检查。正式入口不得绕过批准检查。

## 日志过滤准备检查

已将逐候选日志归到独立 target，并验证关闭它后原基础 pick 日志仍存在、R2记账仍正常。Router policy 的28项本地测试通过。Docker启动器需要转发 `RUST_LOG`，补丁见 [router-log-env.patch](patches/router-log-env.patch)，已通过应用检查。

两份配置已进行仅source的参数检查，见 [resolved-config-check.json](resolved-config-check.json)；没有启动服务。新版release二进制仍需在基线镜像内构建并记录哈希，不能用本机debug产物做性能比较。
