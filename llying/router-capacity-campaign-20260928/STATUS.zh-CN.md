# Campaign 当前状态

更新：2026-09-28 19:51 UTC。用户已授权自主完成方案、持续保存提交，并要求复核方案与代码。

## 当前执行

两次Slurm抢占后，已定位到触发抢占的32047指定了固定8台节点。新申请 **32054** 固定使用该名单之外的 **n04-33 / n05-21**，已于19:26:13 UTC到位，租期至9月29日07:26:13。两台显存空闲、镜像准备已通过，源码哈希与固定基线逐路径一致；RB真实接受率16个答案与P/D各8个rank检查已通过；正式配置预检通过，19:48:54开始884条预热，随后自动进入3600秒C80。

RB将复测 **旧P首次评分＋R1＋P/D会话亲和＋D radix**，作为新节点上后续Triton/容量实验的基线。Fusion开启、IndexShare关闭、有效chunk 4096、MTP模拟3.61、P HiCache1.5、D HiCache关闭均保留。先做真实接受率已知答案/rank覆盖检查，再测C80。

## 已有结果

- **B1 新P首次评分：不纳入后续默认组合。** 输出吞吐−2.45%、平均TTFT+9.82%、P排队+24.68%；见[b1/RESULT.zh-CN.md](b1/RESULT.zh-CN.md)。
- **B2 单独D radix：改善TTFT，尚无吞吐收益。** 输出吞吐−1.04%、平均TTFT−6.45%，D本地复用23.62%。10,295完成、11边界取消、0正式错误；完整限制见[b2/RESULT.zh-CN.md](b2/RESULT.zh-CN.md)。
- **B3 D radix＋D亲和：客户端完成，导出/诊断被抢占打断。** 10,308完成、9取消、0错误；恢复输出TPS约2812.05，较B2+3.06%，平均TTFT+2.47%、p95−4.52%。D原生匹配子集复用约91.8%。缺最终标准导出与完整末尾诊断，不伪装成全部验证通过。见[b3-interrupted/RESULT.zh-CN.md](b3-interrupted/RESULT.zh-CN.md)。
- **R0功能检查通过，未进入性能测试。** 替代allocation 32053又于18:51:32被抢占。此前16个真实接受率已知答案、P/D各8个rank覆盖、会话亲和、room与D本地复用均通过。见[r0-gate-before-preemption/RESULT.zh-CN.md](r0-gate-before-preemption/RESULT.zh-CN.md)。
- 更早D radix检查的24个已知答案与本地复用路径通过；思考文本逐字一致检查未通过，原记录保留。见[b2-gate/RESULT.zh-CN.md](b2-gate/RESULT.zh-CN.md)。
- Router已完成292项单元、26项HTTP、4项ZMQ和14项render验证；多worker统计、探针覆盖、GPU吞吐分母与收尾逻辑均已复核。见[CODE-REVIEW.zh-CN.md](CODE-REVIEW.zh-CN.md)。

## 资源与中断

原31999于18:13:42被抢占。替代32053于18:19取得n10-29/n01-21，旧P显存约17分钟后正常释放，未重置GPU；随后32053也被同一指定节点请求抢占，18:56:35结束。

备用单节点32046曾在18:37:17–18:39:41短暂取得n04-21，用于准备替代P。原P随即正常恢复，因此在仅加载镜像、尚无GPU服务时立即取消，避免长时间空等。当前没有额外第三台分配。

32054和后续第三台均避开当前32047指定的8台，以及长期COMPLETING的n01-25。该约束针对已知冲突，不保证不会有新的抢占。待RB实际启动后重排第三台进入时间；第三台到位后当前正式窗口结束再优先跑2P8+D8、4P4+D8，必要时延后P8D4。

## 后续与恢复入口

RB新节点基线 → B4 Triton P8D8/C80 → 2P8+D8、4P4+D8（同三台24卡）与P8D4/C80 → 汇总保存、资源清理。R2/R4/D HiCache不盲目加入。

机器可读状态：[CURRENT.json](CURRENT.json)。运行目录：`/perf_apps/liyingli/bench_agentx/router-capacity-20260928`。原始请求/trace/采样和所有中断记录保留。prepare_recovery.py --job 32054、run-rb-recovery.sh及allocation watcher已运行，勿重复启动。
