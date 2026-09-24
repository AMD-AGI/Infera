# R1+R4自主实验进度

2026-09-24 18:29 UTC：作业31719已于18:20:10获得n04-33(P=10.235.192.139)、n05-21(D=10.235.192.138)，qos=batch，两节点独占，4小时，排除n04-29。

n05-21残留ATOM服务与已取消作业31718的训练容器。Slurm确认只有本用户31719仍在该节点，已按已分配节点清理授权停止5个确切容器，保留inspect与stop记录。两节点GPU空闲后继续；无GPU驱动重启。D镜像从P复制，ID与历史基线相同。

R1+R4无HiCache smoke于18:26:54开始模型启动。两端初始化中。smoke和正式AIPerf依赖环境均已提前准备完成。正式仍C80/4K、P HiCache开、R2/R3关，对比历史A0/G0，不补跑基线。

276项离线测试通过。已复现单独<think>在glm45 parser变为空正文/空reasoning；实际镜像parser与固定上游源码SHA256一致。仍无法追溯旧3条具体token，待smoke原始响应对照。

运行目录：/perf_apps/liyingli/bench_agentx/r1r4-4k-20260924。最新状态以该目录runs/r1r4-31719-smoke/STATUS为准。

18:50:30 UTC：31719收到抢占通知，EndTime改为18:55:30。正式预热18:49:54才开始，未进入profiling。watchdog停止客户端、D和辅助服务；P停止调用返回未收到exit event错误，后续释放尚未核实。集群PreemptMode=REQUEUE，将继续监控原job自动重排，不重复申请。已通过的无HiCache smoke保留，下一次直接重启正式配置与常规连通性检查，不重跑完整smoke。

18:56 UTC：Slurm已自动REQUEUE，Restarts=1，PENDING(BeginTime)。保持单一申请31719，准备performance-attempt2独立结果目录。

19:10:40 UTC：31719重新分配相同两节点，Restarts=1。P上一轮容器确认18:51:04退出，但显存至19:19:31才全部释放；新尝试19:19:33开始模型启动。CONFIG=performance-attempt2.sh，独立RUN/prefix。客户端准备复用缓存，完整smoke不重跑。基线镜像已归档共享目录，身份见review/baseline-image-identity.json。

19:41:00 UTC：第二次分配在30分钟保护期结束后再次被dcgpu-test训练任务抢占。仍处于首批primer预热，未进入profiling。job31723明确要求包含当前两节点的8节点集合；normal 31720/31721也排在同一8节点集合。下一次在显存空闲后再人工核对高优先级计划窗口，确认有合理完成机会才启动HiCache；不提前把自动模型启动挂在显存等待后面。保持batch和排除n04-29，不通过提高qos绕过要求。

20:02:43 UTC：第三次获得相同两节点，未启动模型。P残留VRAM约96%，D空闲。20:08:45出现新的dcgpu-test作业31724，明确要求相同8节点。虽然预测StartTime显示本job四小时结束时，但前两次该预测没有反映30分钟后触发的抢占。决定不在已知高度可能短窗口内重新开启HiCache；待显存清理并观察资源竞争，必要时重排本job。第三次性能目录只有准备信息，不能记录为第三次已运行实验。

20:13:01 UTC：第三次分配未启动模型。因n04-33退出后仍约96%VRAM占用，主动requeue本job并临时增加排除n04-33；原HIP节点n04-29继续排除。Restarts=3包含这次主动重排，实际正式实验启动次数仍为2，均被抢占于warmup。

20:20:57 UTC：获得n01-33/n02-21。确认残留primus31724已取消且只有本job占有两节点，停止两容器后16卡均0.096%显存。镜像与基线相同，客户端环境复用验证通过。待运行dcgpu-test31725指向另一组节点，不覆盖当前两台。20:26:37第三次正式尝试开始加载模型，CONFIG=performance-attempt3.sh。未重复完整smoke。

20:51:01 UTC：第三次正式尝试仍被抢占于warmup，未进入profiling。此次分配20:20:57，恰在可抢占期20:50:57之后触发。启动时31725并不覆盖当前节点，但随后出现新的dcgpu-test31733，说明预测不能保证后续窗口。决定下一次不再随每个短暂分配立即加载HiCache；先等待当前密集高优先级提交阶段并继续监控。三次已启动正式尝试均未完成预热；smoke结果不重复。

20:57 UTC：Slurm已自动重排，Restarts=4（3次抢占+1次主动重排）。最早分配时间后移至21:30 UTC，保留batch申请并持续观察高优先级任务。下一次实际实验使用performance-attempt4.sh；没有修改模型/负载配置，也不重复smoke。
