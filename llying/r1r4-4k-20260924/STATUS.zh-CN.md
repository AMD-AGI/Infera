# R1+R4自主实验进度

2026-09-24 18:29 UTC：作业31719已于18:20:10获得n04-33(P=10.235.192.139)、n05-21(D=10.235.192.138)，qos=batch，两节点独占，4小时（允许2.5小时回填），排除n04-29。

n05-21残留ATOM服务与已取消作业31718的训练容器。Slurm确认只有本用户31719仍在该节点，已按已分配节点清理授权停止5个确切容器，保留inspect与stop记录。两节点GPU空闲后继续；无GPU驱动重启。D镜像从P复制，ID与历史基线相同。

R1+R4无HiCache smoke于18:26:54开始模型启动。两端初始化中。smoke和正式AIPerf依赖环境均已提前准备完成。正式仍C80/4K、P HiCache开、R2/R3关，对比历史A0/G0，不补跑基线。

276项离线测试通过。已复现单独<think>在glm45 parser变为空正文/空reasoning；实际镜像parser与固定上游源码SHA256一致。仍无法追溯旧3条具体token，待smoke原始响应对照。

运行目录：/perf_apps/liyingli/bench_agentx/r1r4-4k-20260924。最新状态以该目录runs/r1r4-31719-smoke/STATUS为准。
