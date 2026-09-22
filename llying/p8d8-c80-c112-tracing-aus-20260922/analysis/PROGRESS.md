# 实验现场

阶段（18:33 UTC）：诊断镜像与持久化 runner 已就绪，旧服务显存释放中。
用户已授权在两节点执行实验、保存结果、阶段性提交；不需要再次询问。
Prefill stop 命令出现未收到 exit event；禁止GPU reset，需检查实际退出与显存释放。
Decode旧服务已停止，其他用户的 xiaoming-dev 保留。Prefill已确认退出，尚有约60%残留显存；历史上同类回收最终会一次性释放，继续等待，不reset。

两节点诊断镜像一致：`sha256:4f125ff9096f75611fa24caad4c01c3707dd0892251680a6483b99ffaa2a88bb`。
tracing依赖import检查通过。相同基础镜像的AITER内核cache复制到新image-key目录，避免重复编译。
Runner在Prefill后台运行：运行根目录`/perf_apps/liyingli/bench_agentx/p8d8-tracing-aus-20260922`；`driver.pid`、`driver.log`；run为`runs/main-20260922`。断开客户端不会中断。
它依次等待GPU空闲、launch、smoke与关联验收、C80和C112各3600秒；warmup均10/lane。
每60秒回收节点本地诊断与P/D/router日志。collector连续保留。

代码提交：d9458892（诊断补丁）、21af0ee5（runner）、dc4c79ee（关联分析）。
已推送独立远端分支：`llying/p8d8-tracing-aus-20260922`。
测试通过：补丁在实际源码快照可应用/语法解析；admission限频与实际rank字段；synthetic phase isolation、缺失分母和allocation duration。

计划：镜像构建→双节点idle gate→launch+smoke→C80→C112→phase/request关联与瓶颈/均衡分析。
优先获得正式数据，节点剩余时间有限；不可把未完成实验标成成功。
