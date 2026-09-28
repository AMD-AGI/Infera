# Campaign 当前状态

更新：2026-09-28 18:04 UTC。用户已授权自主完成方案、持续保存提交，并要求逐步review方案与代码。

## 当前执行

B3（D radix + D会话亲和）已完成固定884条预热，正式3600秒C80窗口为17:15:08–18:15:08 UTC。P/D引擎复用B2，只切换Router为both亲和。P保持旧首次评分、R1、fusion开启、IndexShare关闭、MTP模拟接受长度3.61均保持。

## 已完成

- **B1 新P首次评分：不纳入后续默认组合。** 输出吞吐−2.45%、平均TTFT+9.82%、P排队+24.68%；匹配请求显示首次派单增加了大段cache miss。代码保留为默认off的实验功能。见[b1/RESULT.zh-CN.md](b1/RESULT.zh-CN.md)。
- **B2 单独D radix：改善TTFT，尚无吞吐收益。** 输出吞吐−1.04%、平均TTFT−6.45%；D本地复用23.62%输入tokens。10,295条完成、11条边界取消、0正式请求错误；另有1条warmup无有效内容记录与取消credit收尾超时，均保留。见[b2/RESULT.zh-CN.md](b2/RESULT.zh-CN.md)。
- **D radix真实接受率功能检查：通过已知答案和本地前缀路径检查。** 24个标准答案正确，命中prefix=10,816、仅清D后prefix=0。思考文本不逐字一致的严格检查未通过，原记录保留；不把小型探测当作完整长上下文准确率证明。见[b2-gate/RESULT.zh-CN.md](b2-gate/RESULT.zh-CN.md)。
- 后续容量切换脚本已复核：修正探针rank覆盖假设与跨worker host-load事件关联，并新增实际GPU吞吐分母检查。见[CODE-REVIEW.zh-CN.md](CODE-REVIEW.zh-CN.md)。
- 292项Router单元、26项HTTP、4项ZMQ、14项render验证通过；多worker placement检查与旧参考分析回归通过。

## 资源

- 原allocation **31999**：n10-29、n03-33，截止2026-09-29 02:52:35 UTC。申请延长至36小时被Slurm拒绝，租期未改变。
- 第三节点 **32046**：单节点/8卡/6小时，当前Pending、不占GPU。已移除未来启动门限，并排除长期处于IDLE+COMPLETING的n01-25；实际作业预测19:25:51 UTC在n04-33启动，只是预测。HIP故障n04-29及原两台仍排除。
- 第三台到位后先准备镜像/缓存；若当前仍有较长正式窗口，可用隔离etcd做启动检查，不接入当前Router。当前正式点完成后优先使用第三台完成三节点点，必要时把P8D4移后。
- allocation监视仅在确认租期丢失、节点归属改变或到期前最后缓冲区处理所属job的campaign容器；查询失败不杀服务。原allocation按用户约定保留，新增节点完成后释放。

## 待完成

B3完整结果与均衡分析 → 选择D组合 → B4 Triton P8D8/C80 → P8D4/C80、2P8+D8/C80、4P4+D8/C80。后两组同三台、24卡，比较不同P并行粒度。若第三台提前到位，调整后面拓扑点的顺序避免闲置。R4、D HiCache不默认启用；R2不盲目叠加。

## 数据与恢复

- 实时机器可读状态：[CURRENT.json](CURRENT.json)。
- 运行目录：`/perf_apps/liyingli/bench_agentx/router-capacity-20260928`。
- 原始请求、trace、采样在各`runs/campaign-*`目录；代码、配置、紧凑结果持续提交本分支。
- B3由已启动的`await-b3-performance.sh`衔接执行，勿重复启动。allocation watcher与第三节点等待/准备进程也已运行。
- 冻结Router binary SHA256：`71750540cb0af0764f72631944f2cf662a9c299799d385524bb373dffb08ca4f`。
