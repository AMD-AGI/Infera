# Campaign 当前状态

更新：2026-09-28 18:30 UTC。用户已授权自主完成方案、持续保存提交，并要求复核方案与代码。

## 当前执行

原allocation 31999于18:13:42被Slurm抢占，18:19:05结束。替代allocation **32053**已于18:19:13到位：P继续使用n10-29，D改用n01-21，租期到9月29日06:19:13。正在准备镜像与正常释放旧P显存；不会重置GPU或停止无关容器。

恢复点R0将直接复测 **旧P首次评分＋R1＋P/D会话亲和＋D radix**，作为新节点上后续Triton/容量实验的基线。Fusion开启、IndexShare关闭、有效chunk 4096、MTP模拟3.61、P HiCache1.5、D HiCache关闭均保留。先做真实接受率下已知答案和rank覆盖检查，再测C80。

## 已有结果

- **B1 新P首次评分：不纳入后续默认组合。** 输出吞吐−2.45%、平均TTFT+9.82%、P排队+24.68%；见[b1/RESULT.zh-CN.md](b1/RESULT.zh-CN.md)。
- **B2 单独D radix：改善TTFT，尚无吞吐收益。** 输出吞吐−1.04%、平均TTFT−6.45%，D本地复用23.62%输入tokens。10,295完成、11边界取消、0正式错误；另有warmup/取消credit收尾情况，见[b2/RESULT.zh-CN.md](b2/RESULT.zh-CN.md)。
- **B3 D radix＋D亲和：客户端完成，导出/诊断被抢占打断。** 容器内客户端完成3600秒发压：10,308完成、9取消、0错误。恢复的输出TPS约2812.05，相较B2+3.06%；平均TTFT+2.47%、p95−4.52%。D原生匹配子集复用约91.8%。缺最终标准导出、末尾完整诊断和引擎身份快照，不能当作完成全部验证的标准归档。见[b3-interrupted/RESULT.zh-CN.md](b3-interrupted/RESULT.zh-CN.md)。
- D radix真实接受率检查：24个已知答案正确、本地prefix命中/清D后未命中路径通过；思考文本逐字一致检查未通过且原记录保留。见[b2-gate/RESULT.zh-CN.md](b2-gate/RESULT.zh-CN.md)。
- Router已完成292项单元、26项HTTP、4项ZMQ和14项render验证；多worker统计、探针覆盖、GPU吞吐分母、进程收尾均已复核。见[CODE-REVIEW.zh-CN.md](CODE-REVIEW.zh-CN.md)。

## 资源安排

第三节点32046目前 **JobHeldUser、未分配GPU**，避免两台恢复期间单独空等。R0开始后根据实际进度重设进入时间；第三台到位后当前正式窗口结束再优先跑2P8+D8、4P4+D8，必要时把P8D4移后。n04-29（HIP stream问题）和n01-25（长期COMPLETING）仍排除。

原分配的清理在抢占后被PAM拒绝，未绕过访问限制。重新取得n10-29后仅清理之前明确记录的本campaign容器；无关sgl.k3、k3.play、dev_primus_wenx、msa_r7_verify不动。

## 后续与恢复入口

R0新节点基线 → B4 Triton P8D8/C80 → 2P8+D8、4P4+D8（同三台24卡）与P8D4/C80 → 汇总保存、释放新增资源。R2/R4/D HiCache不盲目加入。

机器可读状态：[CURRENT.json](CURRENT.json)。运行目录：`/perf_apps/liyingli/bench_agentx/router-capacity-20260928`。原始请求/trace/采样保持在各runs目录，所有中断记录保留。新准备进程prepare_recovery.py和allocation watcher已运行，勿重复启动。
