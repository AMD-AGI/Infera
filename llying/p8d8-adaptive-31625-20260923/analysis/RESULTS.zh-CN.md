# 自主验证结果（持续更新）

更新：2026-09-23 19:52 UTC。当前结论：**G0出现有机制支持的收益；A1在19:46:58被调度抢占，正式窗口仅约3分钟，作废。回滚确认未完成，尚未认定稳定收益或修改默认行为。**

## 1. 已完成的直接对照

同job31644、P n10-29 / D n02-21、16张MI355X、4K C80、相同P/D进程、固定数据revision与seed、完整3600秒profiling。每轮重新建立router/collector、清空逻辑GPU/host cache、统一884条warmup。A0/G0用同一个新router二进制，仅guard释放开关不同。实际P KV/host/D KV容量分别3,143,424 / 4,715,200 / 3,003,264 tokens/rank，均未改变。

| 指标 | A0：P guard随D结束释放 | G0：P HTTP响应体完全drain后释放 | 变化 |
|---|---:|---:|---:|
| 完成请求 | 9,727 | 10,033 | +3.15% |
| total tokens/s/GPU | 20,166.30 | 21,264.18 | **+5.44%** |
| output tokens/s/GPU | 161.66 | 168.70 | **+4.36%** |
| TTFT mean | 9.85979 s | 7.57922 s | **−23.13%** |
| TTFT p50 | 5.50831 s | 4.54819 s | −17.43% |
| TTFT p90 | 22.24993 s | 15.60584 s | **−29.86%** |
| ITL mean | 14.14 ms | 13.48 ms | −4.67% |
| P queue mean | 4,857.12 ms | 2,676.90 ms | −44.89% |
| P forward envelope mean | 3,038.20 ms | 2,880.29 ms | −5.20% |
| 结束边界取消 | 13 | 16 | 保留，不补零 |

两轮profiling导出错误均0、有效导出请求全部P/D关联。G0另有3条warmup空内容导出记录，AIPerf归为InvalidInferenceResultError；不能说整个实验完全无任何无效记录。P/D scheduler PID和startup_time保持不变。GPU所有权监测贯穿正式窗口，A0/G0最大采样间隔17.8/22.4秒，无外部GPU干扰。G0独立engine scrape缺口P15/1792、D19/1792，不补零。

## 2. 机制和工作量

G0的P释放日志全为separated=true。正式窗口被选中P rank的active_blocks mean从13,046降到3,201（约−75.46%），p50从13,035.5降到1,704。它是路由记账，并非物理KV resident或全部rank总和；只观测winner，不能重建所有候选rank的反事实分数。

9,665个共同trace turn中，按预先设定的输入差≤8 tokens且≤0.1%、实际输出长度相同，保留9,091对：

- P queue mean **−43.78%**，TTFT **−22.63%**。
- miss tokens/request **−5.41%**，P forward envelope **−4.23%**。
- host命中tokens/request **+70.06%**；device命中基本不变（−0.06%）。这不支持将改善简单称为GPU缓存命中大增。
- 严格输入/输出长度完全相同的1,790对也显示P queue −36.41%、TTFT −17.32%，但cohort更小且构成不同。

完整G0平均输入+2.24%、平均实际输出+1.18%，闭环轨迹推进有变化。按OSL mismatch反推的请求输出预算，实际/预算比例由96.45%上升至97.60%；这个比值不是从原始wire payload直接读取，也不等同于外部trace的output_expected统计。至少观察到的吞吐收益并非来自更短的实际平均输出。相同trace/输出长度配对进一步支持排队改善，但配对仅取两轮都完成的交集，存在选择效应。

AIPerf聚合server cache hit指标存在counter reset和口径不一致，自动表保留作诊断；结论采用逐请求cache cohort。完整缓存token账本与分层见g0-final/summary.json及g0-vs-a0/。

## 3. 下一步决策

G0通过审计且收益/机制一致，因此19:09:37 UTC选择run_a1：恢复decode模式，继续复用同一P/D并重置缓存/路由器。A1用于检查是否随回滚恢复原表现；单个处理组还不足以宣称稳定收益。

优先把当前有收益的候选验证扎实。P固定host容量扩容至340万tokens/rank的方案及退休/释放流程已准备，但尚未执行；是否投入依A1结果、剩余可完整采集与复核时间决定。D扩容、无metadata的D overlap调参、短请求配额、没有直接证据的host扩容暂不执行。避免把十小时窗口花在低信息扫参和重复HiCache释放上。

## 4. 复现与证据

运行根目录：`/perf_apps/liyingli/bench_agentx/p8d8-adaptive-31625-20260923`。

- A0/G0最终摘要、容量验证、缓存重置、router身份、采样审计分别在`a0-final/`、`g0-final/`。
- 大型逐请求、span、router log和所有权记录保留共享目录，SHA256见各final目录的large-evidence-manifest.json。
- 自动对照：`g0-vs-a0/`；严格及容差配对：`g0-vs-a0-matched/`、`g0-vs-a0-matched-tolerance8/`。
- 路由候选patch、固定二进制hash和镜像源代码一致性记录在本campaign的patches/及analysis/router-*。候选默认关闭，只改变HTTP streaming路径；单元测试已通过，生产Rust源码/默认模式尚未改动。
- 缓存重置的host gauge限制与源码证明见CACHE-RESET-VALIDATION.zh-CN.md；输入cache-bust配对口径见REQUEST-MATCHING.zh-CN.md。

## 抢占更新

job31644 PreemptTime=19:46:58 UTC，extern于19:47:29完成。A1的P/D采样停止、SSH被pam_slurm_adopt拒绝，正式窗口未完成；本地a1-interrupted/INVALID和Slurm证据已保存。远端STATUS仍停留BENCHMARK_C80，不能误当正常进行。D watchdog记录了cleanup_requested，但没有成功停止完成记录；P watchdog无事件文件，不能声称全部容器/显存已清理。此前A0/G0完成窗口与审计均早于抢占，已提交结果保留。原A1预计20:51完成的时间不再适用。

## 最新执行决定：直接运行优化，不重复基线

用户明确要求后续沿用已有基线，不再每次重复A任务。撤销新节点反向两轮及完整重启AB的准备安排；G1使用completion模式、4K C80，在n02-29(P)/n02-33(D)直接启动，20:12:57 UTC进入模型加载。后续仅根据结果选择有信息收益的单变量优化。跨节点比较存在硬件/时间状态差异，分析注明限制，不以此强制追加基线。

job31644重新分配开始19:54:31 UTC，QOS=batch，当前结束00:24:31 UTC。延长至5小时的请求被Slurm权限拒绝，实际仍为4小时30分。新节点默认Slurm客户端指向gpuperf；使用独立client-only配置定向dccs-1334，未修改主机/etc配置，已验证能读取正确job/节点/重启次数。

新节点镜像ID已核对相同；n02-33存在其他ATOM容器但GPU显存/利用率为0，未停止或修改它，运行期继续检查外部GPU使用。新watchdog绑定allocation StartTime和节点身份；抢占时并发对本实验匹配镜像的容器发送5秒stop，减少extern步骤撤销前来不及处理的风险；仍不能保证驱动立即释放显存。启动health gate遇到一次docker inspect超时会在原截止时间内重试，真正退出/OOM仍失败，不通过重启模型来处理探测超时。

## 已完成warmup辅助分析

A1正式段无效，但其884条warmup完成于抢占之前，P/D关联完整。三轮共同810条（首批83、后续727）分析已保存于warmup-controls/。后727条A1的P forward仍比A0低约5.4%，显示有与开关无关的轮次差异；G0的miss/TTFT改善在A1回退后恢复到A0附近，支持路由影响缓存选择的可能性。warmup有效输出全部1 token，不代表正式长Decode驻留，不能据此算出“扣除热身后的净收益”，也不因此追加baseline。

## G1启动阶段再次抢占

第二次分配于20:24:43 UTC被抢占（保护期到20:24:31），extern于20:25:14结束。G1还在首次AITER算子编译，未进入warmup/profiling；INVALID已由watchdog写入共享run，不能当性能结果。P watchdog记录etcd已停止；其他停止没有完成确认，不能声称显存全部释放。job31644再次重排队。账户QOS可用batch/debug/normal/shared-low；本集群preempt/qos规则中batch优先级高于normal，切normal不提供更强保护；更高perf等不在当前账户授权列表。继续只准备优化配置，不重复基线。

## 第三次分配与缓存复用准备

job31644 restart2于20:32:32 UTC实际分配n04-29(P,10.235.192.57)/n04-25(D,10.235.192.131)，早于之前22:29的调度估计，结束01:02:32。G2仅开启现有路由优化，沿用A0历史基线，不启动A任务。

n04-29已有固定诊断镜像和13个完成AITER库，已保存92 MiB共享bundle，并验证hash安装到两节点新的本地cache根/tmp/aiter-jit-100078-g2。未复制build locks或覆盖已有库；bundle/manifest使本次准备不会随节点访问丢失。源库与A0当时复制库的逐文件hash比较另存，不能把不同hash自动视为性能等价。

两节点GPU初查均空闲、无运行容器。网络驱动存在差异：P ionic26.07.9.001/firmware1.117.5-a-147，D ionic26.03.3.001/firmware1.117.5-a-77；旧A0/G0两端为26.03.3.001。镜像和模型相同不能消除这个跨节点网络差异，后续结果明确记录该限制，不追加baseline。

D镜像仍在导入时已安排G2 driver等待固定image ID就绪后启动；PID2757312。当前状态不等于已经开始正式benchmark。

## G2中断与G3同节点恢复

G2在21:03:13被抢占，未开始benchmark。D watchdog在21:03:23确认停止D容器；P无watchdog完成记录。21:10:43 job31644 restart3重新取得相同n04-29/n04-25。实际检查16GPU均约0.28GiB、busy0；P外层容器仍在但GPU进程已不占显存。已按本任务前缀/镜像核对后停止并重命名残留P/collector/etcd和已退出D，不操作其他容器；记录在events/restart3/retire-g2-*.json。

G3于21:16:07 fresh启动，使用相同13库seed安装到新的本地cache，guard=completion，无基线任务。G2启动日志显示P 3,142,720 GPU tokens/rank、host4,714,112，较历史3,143,424/4,715,200约小0.02%；D仍3,003,264。这是相同mem_fraction/ratio在节点上的自动容量标定差异，未作为容量优化修改参数；G3预检查按已观测节点容量验证，结果需保留跨节点容量/驱动限制。

QOS进一步核对：normal/batch没有独立PreemptExemptTime覆盖，继承全局30分钟，两者GraceTime均5分钟；normal优先级更低、被可抢占QOS集合更广。其他分区仅作sbatch --test-only检查，Compute-Multinode返回account/partition不允许，没有实际分配或切换。继续管理员已授权的batch/Compute-DCPT。
