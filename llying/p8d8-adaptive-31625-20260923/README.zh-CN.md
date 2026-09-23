# 自主性能验证工作记录

用户最新要求：**沿用已完成基线，直接运行优化配置和后续单变量优化，不再反复运行基线/回滚任务。** 仅使用管理员已分配的QOS权限。完整结果与历史决策见 [analysis/RESULTS.zh-CN.md](analysis/RESULTS.zh-CN.md)，实时摘要见 [analysis/STATUS.json](analysis/STATUS.json)。

## 当前状态

- 当前没有在运行的benchmark。后续优化验证被节点启动故障及替代作业取消阻塞。
- job31644已在22:53:16 UTC确认n04-33内存释放门槛通过后取消归还；当前无活动benchmark或本轮活动分配。
- 原账户emad/Compute-DCPT/batch的替代job31679于22:33:53获得n01-33/n05-21，22:36:09被UID0取消，未启动模型。取消原因未提供，等待澄清，不重复提交或切账户绕过。
- xiewen12账户仅做过test-only资源检查，尚未得到切换额度的确认，也未实际提交该账户作业。
- G4中n04-33上的Prefill正常就绪；n04-29换成Decode仍复现HIP stream段错误。n04-29已从替代节点请求中排除，没有修改其驱动或底层GPU执行逻辑。
- 16个完成AITER库已保存在共享bundle，后续可恢复到全新的本地cache；镜像、模型、客户端constraint与优化配置均已保存。
- 运行根：`/perf_apps/liyingli/bench_agentx/p8d8-adaptive-31625-20260923`。详细状态见analysis/STATUS.json与RESULTS.zh-CN.md。

## 已有结果

A0/G0在n10-29(P)/n02-21(D)完整完成并通过审计。G0观测到total/GPU +5.44%、output/GPU +4.36%、TTFT mean −23.13%，但不能把单次顺序差异当作完全排除热身/时序影响的收益。

A1在profiling约3分钟时被抢占，正式段作废；其完整warmup已单独分析，见 [warmup-controls/REPORT.zh-CN.md](analysis/warmup-controls/REPORT.zh-CN.md)。G1在另一组节点的首次AITER编译阶段被抢占，没有warmup/profiling结果。两次中断证据分别保存在a1-interrupted/和g1-interrupted/。

## 选择与裁剪

先完成路由优化开启的验证，再按结果和剩余窗口决定固定host容量的P KV扩容。P overlap weight 20→10的建议已暂缓：它可能增加miss/重算，目前缺少亲和过强的直接证据。D扩容、无metadata的D overlap调参和短请求配额不优先执行。P容量方案是3,400,000 GPU tokens/rank、host固定4,715,200，需要重新启动P并验证实际容量/显存余量。

## 复用采集工具

本目录是campaign覆盖层；公共helpers与bench-harness来自同仓库旧tracing目录。analysis/runtime-script-manifest.json记录运行脚本SHA256和仓库源文件；materialize_scripts.py可在空目录重建脚本，拒绝覆盖已有目录或源文件漂移。模型、固定诊断镜像、InferenceX checkout和基础配置仍是独立依赖，其路径/revision见各case验证记录。

新的优化入口是scripts/run_optimized_case.sh：使用CONFIG指定配置，fresh/reuse选择相应入口，只接受completion模式；先等待固定镜像，再采集、分析、审计。通过脚本文件启动并关闭stdin，避免嵌套SSH消耗heredoc后续命令。G4使用fresh启动，但未进入benchmark。

统一采集包括：分配与配置检查、smoke、逻辑cache reset、884条warmup、3600秒C80、独立采样、诊断关联、日志模式与进程身份审计。需要改变引擎参数时重新启动相应服务，不把模型重新加载混入正式计时。

archive_case.py只归档已完成且审计通过的case，较大文件保留共享路径和SHA256。audit必须在P/D仍存活时检查身份，先审计再退休。aiter_cache_bundle.py只保存无构建锁、读取稳定的完成.so，安装目标必须是新的本地cache目录，不覆盖运行中的库。

客户端版本与A1绘图库差异见 [CLIENT-DEPENDENCIES.zh-CN.md](analysis/CLIENT-DEPENDENCIES.zh-CN.md)；后续通过validate_and_pin_client.py将A0精确版本constraint写入runtime.env，约束运行前安装，不改变请求/调度。缓存重置的host gauge限制见 [CACHE-RESET-VALIDATION.zh-CN.md](analysis/CACHE-RESET-VALIDATION.zh-CN.md)。

## 代码与解释边界

路由候选patch位于patches/，默认关闭，只改变HTTP streaming guard生命周期。8个disagg测试与18个policy测试通过；镜像相关Rust源与仓库已核对。生产Rust默认行为未修改。Dynamo源码调研另存于analysis/DYNAMO-SOURCE-AUDIT.zh-CN.md，不宣称已迁移或实测其收益。

A0/G0、跨节点G2和warmup辅助分析的证据等级不同。active_blocks是路由记账，不是物理KV占用；forward envelope包含调度间隔，不是独占GPU时间；warmup输出只有1 token，不能直接换算成正式长Decode的净收益。
