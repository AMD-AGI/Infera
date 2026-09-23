# 实验现场

2026-09-23 恢复：已读取旧会话、核对 git status/log/diff 与共享盘最终状态。
两档正式实验实际已于 09-22 22:15:42 完成；C112 sent=9354、有效完成=9321、
drain取消=33、profiling runner errors=0；9321条有效记录P/D全关联。
总吞吐18203.50 token/s/GPU，较C80下降9.51%；输出吞吐下降2.73%。
最终汇总、客户端结果、完成标记和SHA256已回收到results/main-20260922/final。
新增同时刻8-rank采样分析：C112有922/1773次D采样出现某rank KV≥90%
且另一个<70%，但尚未证明替代rank可以接纳特定阻塞请求。
主要新增等待为P queue；缓存miss工作量增加，不能将全部回落归因于路由。
恢复报告：RECOVERY-AND-FINDINGS.zh-CN.md，包含证据、限制及后续干预计划。
测试：phase/accounting、runtime单位与阶段、同时刻分析缺失/错误样本排除通过。
当前环境SSH和Slurm网络socket被禁止（Operation not permitted），无法进行远程
策略干预或刷新job状态；job31526日志最后01:51:18 UTC仍PENDING，非实时结论。
任务状态：正式采集完成、离线报告已回收并补充；因果/策略收益验证仍待完成。

21:10 UTC：C112 warmup完成，1516.22秒，1243返回，原始审计1240有效+
3无效，0取消。21:09:40.039开始正式profiling，预计22:09:40结束发送。
早期138条profiling记录均有效。服务和collector持续运行，没有清理缓存。

20:45 UTC：C80完成并导出，runner正常切换C112，服务/collector未重启。
C80 profiling发送9662，9651有效完成、11 drain取消、0错误；原始JSONL
只含9651有效profiling记录，取消未导出，已修正分析分母为runner sent。
9651/9651有效记录P/D全部关联。吞吐20115.59101 token/s/GPU，
TTFT p50=5.676s，p90=25.0874s。AIPerf导出出现大量counter-reset警告，
独立endpoint直接采样未发现counter reset；容器ID保持一致，Prefill restart=0。
C112 warmup于20:44:23开始，目标1243；依赖缓存复用成功。
实时采样恢复监督进程仍运行至整体实验结束。C80资源阶段汇总已初步生成。

19:36 UTC：C80 warmup在19:35:44完成，1316.59秒，884返回；原始审计
881有效+3空内容InvalidInferenceResultError，0取消。19:35:44.979开始
正式profiling，3600秒发送窗口预计20:35:44.979结束后drain。
实时资源采样已恢复且门禁复查通过，正式阶段起始前已持续采集约14分钟。
初期126条profiling记录均有效。后续必须完成C80导出、C112及最终关联/资源报告。

19:22 UTC：C80在19:13:48开始warmup（884请求），当前72返回、0 runner错误。
客户端RID注入已在真实请求验证，存在跨rank P/D配对。依赖安装较慢，已让
C112的uv-cache指向C80下载缓存，venv仍隔离。
发现恢复后的采样器拒绝覆盖旧JSONL，原门禁未检查时间而误用旧数据。
已归档旧采样、启动独立采样恢复监督进程，19:22重新通过实时样本验收。
门禁新增样本年龄检查，runner新增采样进程存活检查与恢复归档。
warmup开始至约19:22的engine/node采样缺失，不能用于该段资源归因；
request diagnostics与OTLP/capture连续保留。正式profiling尚未开始。
原始host插桩未覆盖实际Unified Radix Cache，详见OBSERVATION-LIMITS.zh-CN.md。

18:54 UTC：持续接手后发现两次正式负载前的脚本门禁失败并修复。
1. smoke validator 错将内部启动/健康请求纳入关联检查；现按 smoke.json
   的8个RID筛选，并逐请求核验P/D room、独立trace roots与关键阶段。
   8/8通过，多chunk span存在；P/D trace ID不相同，必须用RID关联。
2. inherited config重入导致JSON_MODEL_OVERRIDE_ARGS多一个右括号；实验配置
   在source后显式设置合法JSON，保持现有服务参数不变。
失败证据保存在运行目录recovery/config-validation-failure与driver.log。
通过 --resume-after-smoke 恢复，未重复部署、未执行GPU reset。
采样门禁通过，18:53:19启动C80客户端，目前准备依赖，尚未进入warmup。
report_loop已重新启动。时钟偏差初测保存在snapshot/clock-offset.json；
跨机先后需使用误差界，不能直接比较原始wall-clock span。

恢复检查（18:45 UTC）：已通过 SSH 确认原后台 runner 仍存活（PID 1397749）。
driver.log 显示 18:43:40 完成双节点 idle gate 并启动 collector，18:43:41
进入 LAUNCHING。Prefill/Decode 容器均已启动，日志显示分布式通信初始化中；
尚未通过健康检查、smoke 或开始 C80。未重复启动 runner，未执行 GPU reset。
后续依次进行 smoke/关联验收、C80、C112；正式结果仍待生成与分析。

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
