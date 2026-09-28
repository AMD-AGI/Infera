# Router / cache / backend / capacity campaign

用户于2026-09-28授权按实验方案自主执行，并要求持续保存提交、仔细review代码（方案正确、逻辑自洽、代码正确简洁、注释简洁）。

状态：P评分已实现并review，292项单元测试、26项HTTP、4项ZMQ、14项render验证通过；release已构建，P/D无HiCache smoke引擎正在启动。代码提交19a6c1d2。方案见 ../session-affinity-20260928/followup-analysis/NEXT-EXPERIMENT-PLAN.zh-CN.md。

固定条件：R1、P会话亲和、fusion开启、IndexShare关闭、C80及既有数据/模拟接受率口径。按阶段验证P首次选择、D radix、D亲和、Triton，再比较P8D8、P8D4、2P8D8、4P4D8。D HiCache/R4默认关闭。

资源：allocation31999，两台n03-33/n10-29，原到期2026-09-29 02:52UTC。尝试延长至36小时被Slurm拒绝（Access/permission denied），未改变租期。排除n04-29。第三台在三节点准备完成、两节点还剩约两轮时排队；提前得到则调整顺序优先使用。到期前安排可完成的窗口；若需续接，另申请有效allocation，不依赖未经批准的延时。

完成点逐个提交，记录失败和边界；新实现先离线验证、review，再短时实机，最后正式测量。用户预计约15小时后返回。
