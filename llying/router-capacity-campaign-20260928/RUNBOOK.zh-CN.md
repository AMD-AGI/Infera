# 本轮执行入口

Runtime：`/perf_apps/liyingli/bench_agentx/router-capacity-20260928`。真实状态以CURRENT.json、各run的STATUS和Slurm为准，勿因当前无请求就重复启动。

目前await-b4-placement.sh已经运行，负责RB复核后的B4真实接受率检查、C80与归档。第三节点32056已排队，最早23:00 UTC；准备进程也已运行。基础两台为32054（n04-33/n05-21），到期9月29日07:26:13。

B4完成后，先查看b4/REVIEW.json、同RB的比较与匹配请求。再写runtime/config/selected-backend.sh：DSA_PREFILL_BACKEND、DSA_DECODE_BACKEND、BASELINE_RUN和BASELINE_LABEL。若保留Triton，参考run为campaign-b4-triton；若回退TileLang，参考run为campaign-rb-rebaseline，并在下一次配置生成时加--restart-prefill以避免复用后端不符的P。

第三台PREPARED且B4已复核后，首选B6→B7→B5，避免第三台等候两节点实验。入口如下（示例保留Triton）：

```bash
ROOT=/perf_apps/liyingli/bench_agentx/router-capacity-20260928
bash "$ROOT/scripts/run-capacity-point.sh" b6 "$ROOT/runs/campaign-b4-triton" > "$ROOT/performance-b6.log" 2>&1
bash "$ROOT/scripts/run-capacity-point.sh" b7 "$ROOT/runs/campaign-b6-2p8d8" > "$ROOT/performance-b7.log" 2>&1
bash "$ROOT/scripts/run-capacity-point.sh" b5 "$ROOT/runs/campaign-b7-4p4d8" > "$ROOT/performance-b5.log" 2>&1
```

这些是逐点入口，不能同时启动。每个入口先冻结布局，再做真实答案检查，切换D到模拟3.61、预检、测量、分析与完整复核。生成器拒绝覆盖已有run；若阶段失败，保留现场并从明确的失败阶段续接，不直接重跑整个入口。

B5从B7切换时会停止全部旧P worker，并恢复主节点1P8；第三台出现RETIRED_NODE_FREE且确认无本轮GPU服务后，及时取消其单节点job。不要取消基础两节点32054。清理成功后从allocation-registry.json移除已结束的第三节点，保留其日志与资源记录。

B6/B7用同三台24卡；4P4每个P worker使用4卡、DP4、有效chunk4096、max-running128，保持每rank上限32及P总上限512与2P8相同。D4保持总max-running256，按12张实际使用卡归一化。HIP/SMI编号通过PCI地址映射，NIC映射仍按原配置。

每点复核后更新结果解读和CURRENT/STATUS，运行render_campaign_matrix.sh刷新对照表/图，再提交推送。B3恢复结果与完整归档须继续区分，不能把token吞吐差异直接当请求吞吐提升。
