# 恢复前必读

当前无运行或排队GPU作业。32076/32077已被root取消，所有等待流程和watcher已停止；runtime/STOP_AUTORUN已设置。以下入口是已准备的流程，**不能使用已取消的作业ID直接执行**。

需要先取得有效分配，并更新b6-retry1-real.json/sh、b6-retry1.json/sh、准备记录及run-b6-retry1.sh中的主/额外作业ID。若沿用相同三个节点，B4可继续作为对照；改变主P/D节点则需新的比较基线。

端口保护配置placement-ports.sh必须保留。固定服务端口限于25000–29999，内部自动端口分配器避让25000–32767。新worker需要完成新的etcd注册才能判定就绪，不能只依赖SGLang HTTP健康检查。

B6重试复核通过后，finish-capacity-sequence.sh可衔接B7→B5；其中release_retired_allocation.py会在B5切换已确认第三台GPU清空后释放单节点job。启动前同样必须更新其中旧32077引用，并确认节点、作业所有者和剩余租期。该衔接脚本已做语法检查，但资源取消后未实机执行。

完成这些核对后再移除STOP_AUTORUN、启动allocation watcher与准备/执行流程。任何失败均保留现场记录，不覆盖已有run或伪造review-ready。

---

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
