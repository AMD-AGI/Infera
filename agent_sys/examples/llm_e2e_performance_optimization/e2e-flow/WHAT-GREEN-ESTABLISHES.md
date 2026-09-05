# 一次绿色的全链运行,到底证明了什么

`RUN-PLAN.md` 的 *"What each rung must not be allowed to mean"* 记的是**不能读出
什么**。这份文件记的是另一半:**能读出什么,由哪个 validator、凭哪件产物证明。**

两边都需要,因为失败模式是对称的:一份说「绿了什么都没证明」的文件,和一份说
「绿了就是通了」的文件,一样没用。**目标是让读者能分辨我们哪些绿是承重的。**

> **普查时间戳:2026-09-05T09:30Z。** 下面的调用次数会变;结论的形状不会。

---

## 0. 先说结论

一次绿色的全链运行**确立**:五个阶段的产物能在一张图里逐级传递、每一个 handoff 被
它声明的 validator 评判过、并且其中十一个 validator 曾经真的拒绝过东西。

它**不确立**:选出来的算子是对的(T73)、m4→m5 这条缝是好的(rung 4 从未测过它)、
以及十个从未拒绝过任何东西的 validator 是否真的在判分。

**一句话:它证明这条链能跑通,不证明这条链选得对。** 本轮目标是跑通(用户规矩 7),
所以这正是要交付的东西——**但交付时必须带着这句话,否则「五阶段端到端跑通」会被读成
「流水线可用」。**

---

## 1. 一个 PASS 值多少,取决于三件事

| 问题 | 答不上来时的后果 |
|---|---|
| 这个 validator 被调用过几次? | 一次调用的绿,和一百次调用的绿,不是一个东西 |
| 它**曾经拒绝过**任何东西吗? | 从未拒绝过的 validator,分不清「一直都对」和「根本不判」 |
| 它被**负控制**驱动过吗? | 这是唯一能把上面那两种情况分开的办法 |

第三条是 checkpoint 的工作,前两条可以直接从 run tree 数出来。

---

## 2. 每个 validator 被行使到什么程度(实测,2026-09-05T09:30Z)

数据源 `runs/*/handoffs/*/v*/validation.yaml`,全部 verdict,通过的和失败的都算。

| validator | 调用 | 拒绝过 | 读作 |
|---|---:|---:|---|
| `check_environment` | 375 | 3 | 承重 |
| `check_command_parses` | 166 | **0** | **从未拒绝** |
| `check_bench_result` | 83 | 1 | 承重 |
| `check_deploy_kit` | 63 | 2 | 承重 |
| `check_deploy_serves` | 62 | 7 | 承重 |
| `check_trace_coverage` | 35 | 5 | 承重 |
| `check_kernel_table` | 35 | **0** | **从未拒绝** |
| `check_identity_resolved` | 30 | **0** | **从未拒绝** |
| `check_profiling_evidence` | 30 | **0** | **从未拒绝** |
| `check_worklist_shape` | 30 | **0** | **从未拒绝** |
| `check_workset_shape` | 26 | 4 | 承重 |
| `check_workset_runs` | 26 | 8 | 承重 |
| `check_optimization_shape` | 15 | 5 | 承重 |
| `check_speedup_substantiated` | 15 | 7 | 承重 |
| `check_acceptance` | 12 | **0** | **从未拒绝** |
| `check_bench_report` | 12 | **0** | **从未拒绝** |
| `check_measurement_order` | 12 | 2 | 承重 |
| `check_overlay_applies` | 6 | **0** | **从未拒绝** |
| `check_patch_live` | 6 | **0** | **从未拒绝** |
| `check_no_regression` | 6 | 4 | 承重 |
| `check_packup_shape` | **2** | **0** | **证据最弱的一个** |

**十一个拒绝过,十个从未拒绝过。** 「从未拒绝」不等于「坏了」——`check_command_parses`
调用了 166 次零拒绝,很可能是因为 `mock.sh` 在每次拷贝时就把那个已知缺陷修掉了
(MOCK-MAP J),也就是它守的那条线在上游就被满足了。**但从 run tree 里分不出这两种
情况**,这正是负控制存在的理由。

### 一个 `-noval` 运行确立什么:比你以为的少一整类

**`make_debug_package.py` 生成的树把每个 validator 换成 `check_nothing`**,这是让链
先走起来的正确工具。但在那种模式下:

> **一次 `-noval` 的绿色运行,分不出「走通了」和「走过了四次静默失败」。**

实测,run `20260905T110109-b2e7af`:四个 body 失败,**其中三个被记为 `succeeded`**
——因为 `mock.sh` 已经先把输出拷好了,而**程序 body 的非零退出在输出存在时被丢弃**
(`temp/bugs/2026-09-05-a-failing-program-task-is-recorded-succeeded-when-its-outputs-exist.md`)。
validator 全部是 `check_nothing`,所以没有任何东西反对。**同一机制在 validator 开着
时是被抓住的**——m4 的记录里那次有三个 `strong` 拒绝抓住了它。

**这对本文件 §2 的普查是一个具体的、更强的说明。** 在 `-noval` 下:

- **调用次数照常增长**,每一次都是一个 `check_nothing` 的 PASS;
- **拒绝次数恒为 0**。

于是「调用」和「拒绝」两个数彻底脱钩,**只有拒绝数还有意义**。这比我们推导这条配对
时用的论证更硬:它给出一个真实存在的运行模式,普查数字完好无损地保留下来,同时变得
一文不值。

> **普查和负控制,缺一不可,而且方向相反。**
> **没有普查,没人会去问这个问题**——21 个 validator 全绿看起来一样好。
> **没有负控制,普查会被读成一份控诉**——「十个从未拒绝」听着像十个坏掉的检查,
> 而其中至少有一些只是守的线在上游就被满足了。
> 普查给出**分野**,负控制给出**含义**。引用其中任何一个而不带另一个,都会得出一个
> 比证据更强的结论。

### 2.1 十个「从未拒绝」里,五个已被负控制证明会判分

**checkpoint,commit `3885050`,`work.checkpoint.summary.md`。我第一手核过这份表。**
每一行都是往产物里注入一个已知的破坏、再看 validator 是否拒绝:

| validator | 当时调用次数 | 注入的破坏 | 结果 |
|---|---:|---|---|
| `check_command_parses` | **107** | 语法错误 · 去掉可执行位 · 还原 | **REFUSE / REFUSE / PASS** |
| `check_kernel_table` | 30 | `table.csv` 砍到 3 行,对 `min_rows: 20` | **REFUSE** |
| `check_identity_resolved` | 26 | 所有 resolution 字段置为未解析 | **REFUSE** |
| `check_profiling_evidence` | 26 | 移掉**一个**声明的 part(`kernel_table`) | **REFUSE** |
| `check_worklist_shape` | 26 | ——(见下) | **今天就会拒绝一件真实产物** |

**五个全部承重。那十个 validator 的 236 次调用里,215 次(91 %)现在有一次被演示过
的拒绝背书。** 而且**没有一次崩溃**——每个变异都产出了带诊断的拒绝,不是异常;
本轮反复出现的「崩溃 vs 拒绝」混淆在这里没有发生。

`check_command_parses` 的那一行还带一个 **restore 控制**:破坏后再还原,验证器回到
PASS。**这是把「它在看」和「破坏没送到」分开的那一步**——没有它,一个 PASS 可能只是
注入失败。

**所以上表里这五行应读作「已被证明会判分」,不是「从未拒绝」。** 剩下五个仍然未测,
`check_packup_shape` 是其中最弱的一个。

*(两处数字对不上,都无害:控制是在 107 次调用时跑的,我表里是 166——分母在这一天里
从 23 → 27 → 107 → 166 一路涨,**控制结论不受影响**。`check_kernel_table` 等三行
同理,30/26/26 对我表里的 35/30/30。引用时请写「控制时 N 次,现在 M 次」。)*

`check_packup_shape` 是**全包里证据最弱的**:两次调用、零拒绝、且据 checkpoint 报告
在负控制扫描里无法驱动。它评的又恰好是**整条流程唯一的对外产物**。一个来自它的绿,
含金量远低于一个来自 `check_workset_runs`(26 次调用、8 次拒绝)的绿,文件应当这么说。

*(负控制的结论是 checkpoint 的工作,我转述而非复现。我第一手核过的是上表的调用与
拒绝次数,以及 `check_packup_shape` 的 2/0。leader 转述时说的是「一次调用」,现在
是两次——差别不影响结论。)*

---

## 3. 分阶段:绿了确立什么

### m1 deploy —— 确立得最扎实的一个阶段

**确立:** kit 是**可被别人跑起来**的。`check_deploy_serves`(62 次调用、7 次拒绝)
拿 kit 自己的 `scripts/deploy.sh`,在**不同的 run tag、端口段和 work root** 下重新
部署一遍,跑 11 个探针和一次 1k/1k、并发 16、三分钟的负载,再拆掉。**一个只对作者
有效的 kit 会在这里失败。** 它的拒绝清单里有真实条目:`environment.md` 没有渲染
`fixed.image`、模型以文件系统路径的名字对外服务、记录内不变式 `gpu_devices` 9 项对
`gpu_count` 8。

**不确立:** 一个模型、一台节点、一次。且 `check_deploy_serves` **判形状不判答案**
——completion 探针只要求 `finish_reason: stop`、`content` 非空、模型 id 不是路径,
所以一个把 526 字符思维链放进答案位置的部署**照样通过**(`todo.md` T21/T28)。

### m2 profiling —— 机制最强的一个 PASS

**确立:** trace 里**真的有东西**。`check_trace_coverage`(35 次调用、5 次拒绝)
**自己解压并解析 trace**,按 rank 数 GPU kernel 事件,再和 producer 写的 manifest
对账——它的 PASS 是**对 trace 内容的读取**,不是对「抓取命令退出码为 0」的读取。
一个开在空转调度器上的窗口会产生八个格式完美、内容为空的 trace 文件,在磁盘上和一次
好的抓取无法区分,而这条检查是唯一能看进去的。

`check_bench_result` 另外加了一条**绝对门槛**:graph ceiling 必须不低于负载实际达到
的并发(CONTRACT §4.6),因为两臂可以共享同一个故障,任何比较都看不见它。

**不确立:** `check_profiling_evidence`(30 次调用、**0 次拒绝**)是跨部件唯一的
一致性检查,而它从未拒绝过任何东西。它守的东西是真的——两条线是否同一次部署——
但我们没有证据说明它会在该拒绝时拒绝。

### m3 analysis —— 唯一一个「重新测量」的信任链

**确立:** workset 印出来的数字是**在这块硬件上被复现过的**。`check_workset_runs`
(26 次调用、8 次拒绝,`cost: gpu_hours`)真的重跑 workset 自己的入口,5 组 ×10 次、
rsd ≤0.10,并与记录对账。**这是全包唯一的信任链根**:M4.3.5 允许 m4 直接采信 workset
的 baseline,其安全性完全建立在这一条上。

**不确立:** **选出来的算子是对的。** T73:Triton pattern 在 `buckets:` 下的那一份
漏匹配时是**从候选池里剔除**,`rank` 根本看不见——实测候选池 18 → 25 个 kernel、
10.74 % → 13.96 % 的 GPU 时间,最大的新增者 2.350 % 会排到第 2。**这个缺陷在 `rank`
的输入侧,所以第一条绿链是在一个被扭曲的池子上绿的。** 决定不修是 leader 的决定
(本轮目标是跑通),但绿了不代表选择被验证过。

### m4 kernel opt —— 承重,但这一级测不到它的下游

**确立:** 加速比是**在 workset 自己的前提下、用 workset 自己的入口**测出来的。
`check_speedup_substantiated`(15 次调用、7 次拒绝)在硬件或优化前提不符时 **abort**
而不是换分母;`check_optimization_shape`(15 次、5 次拒绝)把 `apply.integration_point`
和 workset 声明的 `edit_target` 逐字段对齐。两者都真的拒绝过东西。

**不确立:** **rung 4 绿测不到 m4 → m5 这条缝。** `mock_m5.sh` 读取零个
`AGENT_SYS_INPUT_*`;`apply_patch` 的 mock 分支在读 `--kernel-optimization` 那一行
**之前**就 `exec` 掉了。所以被 mock 的 m5 从不看真实 m4 的产物,这条缝**第一次被行使
就是 rung 5**。详见 `RUN-PLAN.md`「Two rungs license less than they appear to」。

### m5 integration —— 三条代码路径的**首次**行使

**确立:** 补丁**真的在跑**(不是只被挂上)。`check_patch_live` 在运行中的容器里重新
hash,并自 2026-09-04 起**默认要求运行期 marker**——因为实测过一个 in-container hash
完全正确、`.pyc` 当分钟编译的 overlay,测出来和 stock 一模一样;静态证据分不清
「挂上了没执行」和「执行了没效果」。

`check_no_regression`(6 次调用、4 次拒绝)**不读报告的 `verdict` 字段**,从原始数字
重算整个论证,结论不一致即失败——哪怕它自己算出来是「接受」。

**不确立,而且这是 rung 5 特有的:三条路径从未被行使过。**

| 路径 | 为什么从未跑过 |
|---|---|
| `adhoc.json` | 封存语料里没有;rungs 0–4 是 `m5_agent=runner`,**没有 agent 能生成 case**。rung 5 是第一次 `check_acceptance` 的 ad-hoc 臂真的评到东西 |
| `e2e_packup` | 没有封存来源,mock 是合成的。所以 `check_packup_shape`(2 次调用、0 次拒绝)只评过合成产物 |
| `integration_report` 的裁决 | 封存的那份带的是**被拒绝**的裁决;rung 5 是这条流程第一次用自己的数字算出一个裁决 |

`check_patch_live` 和 `check_overlay_applies` 各只被调用过 **6** 次且从未拒绝,
`check_acceptance` 12 次从未拒绝——m5 的证据面是全包最薄的,而它是最后一关。

*(m5 这一节的三条路径来自 MOCK-MAP (D″)(F)(E) 加我自己第一手做的 `adhoc_cases`
分析。我**没有**通读 `mock_m5.sh` 的 `arms` 路径,所以不声称这是「真实 m5 首次产出」
的完整清单。)*

---

## 4. 交付时该怎么说

**可以说的:** 五个阶段的产物在一张图里逐级传递并全部通过了各自的 validator;其中
十一个 validator 在此前的调试中真的拒绝过东西;m1 的 kit 被独立重新部署验证过;
m2 的 trace 被解析验证过内容;m3 的 workset 数字在本硬件上被重新测量过。

**必须一起说的:** 本轮目标是跑通,不是性能提升(用户规矩 7);算子选择尚未被验证
(T73);m4 → m5 这条缝到 rung 5 才第一次被行使;m5 的三条路径是首次运行;十个
validator 至今从未拒绝过任何东西,其中最弱的 `check_packup_shape` 恰好评的是唯一
对外产物。

**不能说的:** 「流水线可用」「选出的 kernel 是值得优化的」「validator 都验过了」。

---

## 5. 这份文件的边界

- 上表是我第一手数出来的,时间戳在页首。
- 负控制的「可驱动/承重」结论是 **checkpoint** 的,我转述。
- m5 的三条路径部分来自 MOCK-MAP 的章节而非通读 `mock_m5.sh`,已在该节标注。
- T73 的数字是 **m3** 的测量,我核过机制(`taxonomy.py` 对未匹配符号返回
  `routable: False`),**没有复现那些数字**。
