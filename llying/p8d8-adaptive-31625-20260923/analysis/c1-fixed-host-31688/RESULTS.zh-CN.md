# C1 固定 host 的 P GPU KV 扩容：完成及复核

2026-09-24。结论：本轮完成且审计通过，但未证明端到端收益；停止继续放大容量。

P=n02-29（AMDGPU 6.14.14），D=n02-33（6.16.6），job31688。两节点 ionic 26.03.3.001 / firmware 1.117.5-a-77。P GPU KV 3,143,424→3,400,000 tokens/rank，host 固定4,715,200，路由保持 decode guard 模式，4K C80。

## 完整结果

| 指标 | 历史 A0 | C1 | 变化 |
|---|---:|---:|---:|
| 成功请求 | 9,727 | 9,774 | +0.48% |
| total tokens/s/GPU | 20,166 | 20,702 | +2.66% |
| output tokens/s/GPU | 161.66 | 161.86 | +0.12% |
| TTFT mean (s) | 9.8598 | 10.370 | +5.17% |
| TTFT p90 (s) | 22.250 | 24.017 | +7.94% |
| P queue mean (ms) | 4,857.1 | 5,457.6 | +12.36% |
| P forward envelope mean (ms) | 3,038.2 | 2,988.1 | -1.65% |
| 实际平均输入 tokens | 119,430（约） | 121,870（约） | +2.04% |
| 实际平均输出 tokens | 965.11 | 960.33 | -0.50% |

total throughput 含命中输入，且闭环工作量改变。output throughput 基本持平、TTFT和排队更差，不将 total +2.66% 判为已验证收益。跨节点/驱动及轮次差异限制因果解释。

## 配对与缓存

输入差≤8 tokens且≤0.1%、相同实际输出长度的8,897对：device tokens/request +0.55%，host -66.31%，miss -3.99%，P forward -1.24%，P queue +13.93%，TTFT +6.08%。

完全相同输入/输出长度的2,634对：device +0.36%，host -92.87%，miss反而+7.19%，P queue +30.24%，TTFT +16.21%。严格子集更小、构成不同；不能把请求级缓存改善称为普遍稳定收益。两种配对均显示排队/TTFT没有改善。交集有选择效应，不用于推导总体吞吐。

完整请求级miss rate由4.9919%降至4.6364%，但不同集合的miss方向不一致。AIPerf聚合cache/counter有reset告警，保留诊断，不用作缓存结论。两端scheduler PID/startup_time均保持不变，不能将这些告警当成实际服务重启证据。

## 有效性及资源

- 02:59:25.702 UTC开始正式窗口；03:59:25.704停止新增请求，04:00:05.703完成收尾；04:05:45自动分析完成。
- warmup 884条成功、0错误；正式9,774条成功、0错误，收尾取消22条。全部9,774条正式导出记录均关联P/D。
- 配置、容量、逻辑缓存清空、模型元数据、路由模式、服务PID及所有权覆盖等14项审计全部通过。
- 所有权采样最大间隔19.37秒，没有外部GPU干扰。正式节点采样P/D各720次；显存采样峰值P95%、D87%。采样不能排除瞬时峰值。
- 独立engine scrape正式P9/1794、D11/1794缺失，不能补零；此前进度中的“采样正常”指节点/GPU所有权监控，并非所有HTTP指标抓取均无缺失。
- 全部启动服务日志检索未找到 `OutOfMemoryError`、`out of memory`、`Fatal Python error` 或非零 `#retracted-req`。

## 后续决定

停止更大容量试探，不修改默认容量。当前P/D稳定且已得到同节点C1完整记录，优先复用本轮服务开展C1G1：GPU/host容量不变，清空逻辑缓存、重建router/collector，仅将guard模式切到completion，直接与C1对照。这检验路由在当前容量下的增量作用，可避免重启和新增跨节点混杂；不代表C1容量被认定有效。20→10 overlap试探继续暂缓。

原始共享run：`/perf_apps/liyingli/bench_agentx/p8d8-adaptive-31625-20260923/runs/c1-fixed-host-31688`。小型证据已复制到本目录 `evidence/`；两套配对见 `matched-tolerance8/`、`matched-exact/`。大型日志、trace和逐请求记录保留共享run。
