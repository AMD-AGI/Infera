# 资源恢复后的执行入口

当前没有运行或排队GPU作业。32076/32077已被root取消，等待流程和watcher已停止，runtime/STOP_AUTORUN已设置。**不能使用已取消的作业ID直接执行。**

Runtime：`/perf_apps/liyingli/bench_agentx/router-capacity-20260928`。

1. 取得新的有效分配。优先保持P=n04-33、D=n05-21、额外P=n05-29；改变主P/D节点时需重新确定比较基线。
2. 更新b6-retry1-real.json/sh、b6-retry1.json/sh、run-b6-retry1.sh及准备记录中的主/额外作业ID。准备流程分别是`prepare_recovery.py --job NEW_MAIN_JOB`和`prepare_extra_node.py --job NEW_EXTRA_JOB`。
3. 核对所有者、节点、剩余租期及准备记录。将STOP_AUTORUN与旧capacity-sequence-failed.txt移到历史记录目录，再启动allocation watcher和run-b6-retry1.sh。已有失败目录不得覆盖。
4. B6完整复核后，再运行B7、B5。可使用finish-capacity-sequence.sh，但必须先更新其中额外job的旧32077引用。该衔接脚本已做语法检查，资源取消后尚未实机执行。

后两点的独立入口：

```bash
ROOT=/perf_apps/liyingli/bench_agentx/router-capacity-20260928
bash "$ROOT/scripts/run-capacity-point.sh" b7 "$ROOT/runs/campaign-b6-2p8d8-retry1" > "$ROOT/performance-b7.log" 2>&1
bash "$ROOT/scripts/run-capacity-point.sh" b5 "$ROOT/runs/campaign-b7-4p4d8" > "$ROOT/performance-b5.log" 2>&1
```

不得并行启动这些点。每点先冻结布局、真实答案检查，再切换D到模拟3.61，预检、测量、分析及完整复核。生成器拒绝覆盖已有run；失败时从明确的失败阶段续接，不直接重跑整个入口。

B5从B7切换时，第三台出现RETIRED_NODE_FREE并确认没有本轮GPU服务后，及时释放其单节点job。release_retired_allocation.py会核对节点、job所有者和已清空的过渡记录，再取消该job；不得取消基础两节点job。清理后保留日志与资源记录。

端口保护配置placement-ports.sh必须保留：固定服务端口限于25000–29999，内部自动端口分配器避让25000–32767。新worker完成新的etcd注册后才能判定就绪，不能只依赖HTTP健康检查。

B6/B7须使用同三台24卡。4P4每worker TP4/DP4、有效chunk4096、max-running128，保持每rank上限32及P总上限512与2P8相同。D4保持总max-running256，按12张实际使用卡归一化。HIP/SMI通过PCI映射，NIC选择仍按原配置。

每点复核后更新解读和CURRENT/STATUS，运行render_campaign_matrix.sh刷新表/图，再提交推送。B3恢复结果与完整归档继续区分，不能把token吞吐差异直接当请求吞吐提升。
