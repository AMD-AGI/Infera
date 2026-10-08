# SGLang PD 与 Router 实验总结：配置、性能变化与原因

覆盖 2026-09-22 至 2026-09-29 已保存的本项目实验，2026-10-08 重新核对汇总。本报告包含 15 个完成或恢复的性能测量点，以及尚未取得数据的容量方案；功能 smoke、启动失败和他人的实验不冒充本项目性能结果。

**目前最有依据保留的是：P 完成后释放路由记账、P 会话绑定，以及在 D radix 下使用 D 会话绑定；DSA backend 使用 Triton。** 前两项主要改善 P 排队与缓存局部性；D radix＋D 绑定最明确的效果是 D 本地复用和接收流量下降，尚不能宣称同等幅度的端到端吞吐提升；Triton 的同节点测试同时改善吞吐和 TTFT。新 P 首次评分、R4、8K chunk、单纯扩大 P KV 和 C112 均未给出足够收益依据。

本报告不是“所有实验已完成”的验收单：P8+D4、2P8+D8、4P4+D8 仍无正式性能数据。单轮测量没有重复运行置信区间，报告中的“正收益”表示本轮观测且有相应诊断支持，不代表跨负载稳定加速。

## 配置与代号说明

- **P** 是 prefill，**D** 是 decode。已测性能点都是 1 个 P worker 用 8 卡、1 个 D worker 用 8 卡，共 16 张 MI355X；引擎配置 TP8/DP8，不能将 8 个 DP rank 理解为 8 个独立 TP8 worker。
- 基本配方：同一项目的 GLM-5.2-MXFP4 / AgentX 回放，MTP 模拟接受长度 3.61。除“并发增至112”和“chunk增至8K”外，都是 C80、有效 chunk=4096；正式发送窗口 3600 秒，吞吐采用原客户端含收尾的统计口径，不能直接用完成数除以3600替换。
- P HiCache 保留历史配置（host ratio 1.5，write_through / kernel / page_first）；D HiCache 关闭。通常 P GPU KV=3,143,424、host=4,715,200、D GPU KV=3,003,264 tokens/rank。仅两行“扩容”将 P GPU KV 提至3,400,000，host不变。各源配置是容量微小标定差异的最终依据。
- **P完成释放（R1）**：P 响应体完成后释放 P 的路由预约；原行为要等 D 完成。这改变路由记账，不是主动清除物理 KV。
- **有效工作量评分（R4）**：按已预约有效输入与本次有效输入评分的实验策略。**Dynamo式首次评分（B1）**是另一个策略：融合 GPU/host 命中、在途有效工作和 active blocks；两者不同，后续均关闭。
- **P绑定**：同一会话后续 turn 保持同一 P worker/rank。**P/D绑定**：P 和 D 分别保持各自目标，不要求 P 和 D 的 rank 编号相同。D radix 与 D 绑定也是两个独立开关。
- **B2**是在原 P 评分＋R1＋P绑定上只开 D radix，**没有沿用 B1 新评分**。**B3**再加 D 绑定。**RB**是 B3 组合换节点后重新测的 TileLang 基线。**B4**在 RB 上只把 P/D 的 DSA backend 切成 Triton；不是替换全部算子实现。Fusion保持开启，IndexShare关闭。
- 配置省略项按以下规则读取：除Triton行外均为TileLang；只有明确写出P/D绑定的行才启用对应会话绑定，只有明确写出D radix开的行才启用D radix；没有写P完成释放的早期行沿用等D结束释放。R2/R3不在这些已测变体中。
- 表格按对照关系排列，不保证执行先后；8K 的同节点4K参照实际在8K之后补测。P/D节点详见附表；跨节点项已经标注，不能把表格第一行与最后一行的差直接称作累计优化收益。

## 性能总表

主表按用户指定指标排列：token/s/GPU、TTFT p50/p90、ITL p50/p90、P miss、D本地复用、Interactivity p50/p90。token/s/GPU同时列出 **Total（含缓存命中的逻辑输入＋输出）** 和 **Output（仅生成输出）**，避免将逻辑输入吞吐当成GPU计算吞吐。TTFT/ITL越低越好，Interactivity越高越好。

每格为 **实测值（相对“比较基线”的变化）**。P miss为请求级未命中token占输入比例，D本地复用为D allocator观测前缀token占输入比例；这些比例的变化用pp（百分点），其他指标用相对百分比。

**Interactivity沿用InferenceX原定义：`intvty_p50 = 1 / ITL_p50(s)`，`intvty_p90 = 1 / ITL_p90(s)`，单位token/s/user。** 它不是对逐请求token/s取同名分位数；因此p90可以小于p50。正式结果使用源文件保存的未提前舍入数值，不能从表中舍入后的ITL精确反推。它只反映生成阶段，不含TTFT；E2E-normalized Interactivity另外保留在CSV。

表格较宽，可打开同目录 `REPORT.html` 横向滚动或使用 `metrics.csv`。随后按 **有效P/D batch → D KV占用 → 排队/重算与完成工作量** 展开对比，每张表使用相同实验名称。batch和KV统计另存 `dimensions.csv` 与 `dimensions.json`，后者附原始日志/采样哈希、窗口、样本数、缺失和rank覆盖。

以下三类统计不可互换：主表是客户端完成请求统计；资源表是固定3600秒发送窗口内约2秒间隔的采样统计；batch日志是实际执行时的本地batch统计，排除了无batch的空闲时间。B3的客户端分位数从完整原始请求恢复，并用B2官方导出校验算法；B3资源与日志仍缺少末尾，不能补零。
