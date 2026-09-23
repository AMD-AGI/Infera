# 自主性能验证工作记录

用户授权在约十小时内自主选择、裁剪和增加实验，置信度优先，可在验证缓存重置后复用服务，随时保存和提交。

## 当前分配和运行

- 原 job 31625 于 14:50:32 UTC 被抢占，复用尝试尚未执行。
- 替代 job 31641 的节点有其他用户 GPU 工作，未运行负载，已释放并排除这两节点。
- 当前 job 31644：P n10-29 / 10.235.192.140，D n02-21 / 10.235.192.128。
- 运行根：`/perf_apps/liyingli/bench_agentx/p8d8-adaptive-31625-20260923`。
- A0：`runs/a0-guard-decode`，原始 P guard 释放时机，新的共同路由二进制。
- G0：`runs/g0-guard-completion`，P HTTP 响应体完整 drain 后释放 P guard。
- G0 后先分析决定是否运行 A1 回退确认，或结束该方向转向其他候选。决策文件 `guard-decision.json` 由分析后填写，不自动把失败数据当收益。

当前状态见 analysis/STATUS.json；远端实时状态见 `guard-campaign-status.json`、各 run 的 `STATUS` 和 `c80/runner.log`。

## 已完成准备

- 诊断镜像的 Rust 源码已从镜像 layer 提取，与仓库相关源码逐字节核对。
- 路由候选只改变 HTTP 流式 P/D 的 guard 生命周期，默认关闭。控制和处理使用相同二进制 SHA256，均有相同的 P 完成观测日志。
- 8 个 disagg 测试（含新生命周期/错误/取消用例）和 18 个 policy 测试通过。二进制已在原诊断镜像内执行 --help 校验。
- 新的按 byte cursor 采集边界可隔离复用引擎的各轮日志；已验证不混入旧数据及不保留未完成尾行。
- 缓存重置采用已核验实现的 8 rank 完成确认、GPU/队列为空、容量和进程身份不变。Host-used idle gauge 陈旧的问题单独记录，见 CACHE-RESET-VALIDATION。
- 完整 case 审计包含配置、模型元数据、PID、日志模式、P/D 关联覆盖率、正式错误和 GPU 隔离证据。
- 本节点 allocation watchdog 在抢占或租期结束前40分钟清理本任务自己的容器；镜像ID/名字前缀不符则拒绝处理。它不能保证被SIGKILL时仍可执行，也不能消除驱动自身的显存回收延迟。

## 选择与裁剪

D 容量仅影响少量长尾，暂不优先；D overlap 没有可用缓存块信息，不做权重扫参；短请求份额/公平性按已有决定排除。新增 G 候选基于已确认的生命周期差异，可省去多次 P 重启。

P 容量候选准备为 3,400,000 GPU tokens/rank，host 固定为 4,715,200 tokens/rank。max_total_tokens 是容量 cap，0.90 fraction 提供预算，ratio 用于补偿 host 容量。实际容量及其余字段必须通过检查才允许发压。该方向是否执行取决于 G 的结果和剩余时间。

## 解释边界

A0/G0/A1用于区分处理收益与复用/JIT/时间漂移。单次差异不提供置信区间。P HTTP 完成到客户端完成的正时间差只是过长记账的请求生命周期代理，不等于 distinct-block active load，也不能直接换成吞吐收益。

Dynamo 已固定 commit 取得源码；研究核验见 analysis/DYNAMO-SOURCE-AUDIT.zh-CN.md，不宣称已迁移或测得 Dynamo 收益。
