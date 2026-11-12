# 交接：复盘讨论（2026-11-12 08:30 UTC）

新会话用来和用户讨论总体复盘与下一阶段计划。本目录 = `/home/liyingli/code/bench_agentx/Infera/llying/dp-graph-variant-fix-aus-20261103/`。

## 1. 目标（用户 11-12 07:16 说明）

让 GLM-5.2（MI355X，SGLang + Mooncake，PD 分离）在 InferenceX 的 Pareto 曲线（每 GPU 吞吐对交互度）上尽量接近或超过 NV Dynamo + SGLang 的 GB300。
- 提高最高吞吐是一条路；降低 ITL、提高交互度 p90、让曲线右移也是目标。
- 所以 decode 侧的工作有效，prefill 决定曲线最右端那个点。

## 2. 阅读顺序

1. `PLAN-REVIEW-20261112.zh-CN.md`：按 Pareto 目标逐项复核 11-09 的 kernel 计划，列出已做 / 未做、重新排序和建议顺序。**讨论从这里开始。**
2. `PARETO-TP8-DECODE-20261112.zh-CN.md`：TP8 decode 能往哪里发力。MoE 只有 3.3–3.8 TB/s，TP4 的 KF 是 5.81 TB/s，每步约 3–6 ms 的空间。
3. `REVIEW-20261112.zh-CN.md`：总体复盘。开头已修正前提；其中"decode 对计分没用"只对 PP8 最高吞吐那个点成立。另有优化盘点、测量问题和执行问题。
4. `REPORT-20261112.zh-CN.md`：11-11 15:12 ～ 11-12 03:12 的工作汇报。
5. 背景：
   - `MEGATP-SUMMARY-20261111.zh-CN.md`：MegaMoeTP 的实验与结论；
   - `BOTTLENECK-PP8-20261109.zh-CN.md`：PP8 受 prefill 限制的证据；
   - `TP8-KERNEL-HANDOFF.zh-CN.md`：TP4 和 TP8 的 MoE 差距，以及"InferenceX 不允许 overlay"这条约束；
   - `KERNEL-PLAN-20261109.zh-CN.md`：原计划。
6. 原始记录：`KERNEL-PROGRESS-20261109.zh-CN.md` 第 15 节（11-11 15:51 之后的全部实测，按时间顺序）。

## 3. 当前状态

- **作业：**
  - 两节点 holder 35609：每次申请 6 h，被抢占后自动重排，已重启 10 次以上。11-12 07:43 在 n01-29 + n02-29 开始，最晚到 13:43。
  - 队列 `/apps/liyingli/holder/queue/` 里没有本会话的条目；别的会话也用这个 holder（mc4540、sgl48302 等），不要动它们的条目。
  - 没有单节点作业：35486 于 11-12 02:47 被 root 取消，用户选择暂不重新提交。
- **已就绪、待端到端的 TP8 实验配置**（`/perf_apps/liyingli/bench_agentx/opt-aus-20261103/pkg/config/`，都基于 `arm-C-1p1d-pp8-megatp-mt4q3`）：
  - `-mt4q3`：MegaMoeTP q3；
  - `-mt4q3-sparse2g`：sparse MLA 拆两段；
  - `-mt4q3-kvcap`：KV 上限；
  - `-mt4q3-gemmtab`：调优表补行；
  - `-mt4q3-gemmmerge`：GEMM 合并（基于 gemmtab）；
  - 另有 TP4 的 `arm-C-prod-ptpcfix-nohip-megatp-mt2q3`。
  - GEMM 合并原型的机制测试已全部通过；起引擎的精度门禁还没跑。
- **查出的漏项（还没改）：**K5（`DSA_LPT_GROUP_ROWS=4 DSA_LPT_NG2=32`）只在 TP4 的实验配置里，TP8 的配置都没有。只改配置，TP8 每步约 −0.4 ~ −0.6 ms。
- **工具：**
  - `opt-aus-20261103/scripts/ana/step_cats.py`：按类别统计 decode 每步的 kernel 时间；
  - `kopt-20261109/bench/` 下的各个 microbench；
  - 合成负载 trace 在 `/perf_apps/liyingli/vfix-logs/opt/trace/`。

## 4. 待用户决定

1. 是否以 TP8 MoE 为下一阶段主线（GEMM 效率提到 ≥ 5 TB/s，每步约 −3 ~ −6 ms）。
2. 上游交付（aiter / SGLang PR）由谁做、何时开始；不交付的话，InferenceX 上不算数。
3. dense FP8 是否值得投入（要改精度、过精度评测）。
4. 是否提供 GB300 曲线上要追的具体点（交互度、吞吐）。
5. 之前悬而未决的：KV 上限；MegaMoeTP 接入正式代码（等端到端结果）；单节点作业；prefill profile。
6. **6.19 驱动节点上 PD 起栈崩溃：没有解决，只是绕开。**详见 `REPORT-20261110.zh-CN.md` 第六节。
   - 现象：Mooncake 把 KV 显存注册给 ionic RDMA 网卡后，创建 HIP stream 时段错误。
   - 已排除："6.19 重复记账"的推断——6.16 上同样的小模型测试，行为和门槛都一样（0.5 正常、0.65 崩），区分不出驱动。
   - ROCm 10 镜像加小模型在 6.19 上也崩。
   - 6.19 特有的问题只在生产配置上出现。下一步本应在 6.19 节点上用生产配置做显存探测（memprobe），一直没做。
   - 现在靠条目开头的驱动检查，自动排除 6.19 节点。如果集群全部升级到 6.19，我们就起不来了。
   - 是否上报集群管理员，待用户决定。

## 5. 必须遵守的规则（用户历次要求）

- **语言和文件：**用中文回答；答案写成报告文件并给出路径。
- **提交记录：**进度、数据和分析都要记录，再用 `COMMIT="..." bash /perf_apps/liyingli/bench_agentx/vfix-aus-20261103/scripts/sync_records.sh` 提交。新文件要先在本目录 `git add`。编辑和提交不要并行执行，提交后用 `git status` 核对。
- **登录节点：**有约 4 GB 内存上限，不要在上面跑耗内存的任务。trace 分析、编译放到计算节点：`srun --jobid=<作业> --overlap -w <节点>`，或者写成 holder 条目。
- **别人的东西不要动：**srt-slurm CI 作业，以及别的会话的条目和作业（如 `llying-e1ab`、`llying-srt630-dev`、`llying-sgl-utcpu`、`llying-gsp*`、`llying-mc4540*`）。
- **作业：**单节点作业结束后不要释放；最多同时 1 个单节点 + 1 个两节点作业。
- **容器：**占着 GPU 显存的外来容器要停掉；不占显存的不要动。
- **比较：**只在同一对节点上比较分数，不同节点对之间 PP8 吞吐能差 4%。
- **表达：**不要用自造的代号或标签；必须用时先解释清楚。避免重复实验。
- **InferenceX：**不允许 overlay / 运行时替换函数，最终成果要以 aiter / SGLang PR 交付。
- **长时间无人时：**挂定时心跳；队列空了就按约定的清单补条目，不要让节点空闲。
