# 自主验证结果（持续更新）

更新：2026-09-23 19:12 UTC。当前结论：**G0出现有机制支持的收益，已启动A1回滚，尚未认定稳定收益或修改默认行为。**

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
