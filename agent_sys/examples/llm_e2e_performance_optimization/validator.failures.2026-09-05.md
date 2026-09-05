# validator 失败记录 — 全量收割

> **普查时间戳:2026-09-05T07:30Z。** 先读这一行——**本文件按它自己的记录，一小时
> 内就会过时**:06:57Z 那一版写下时是 45 条,33 分钟后重数就是 46 条。任何引用本文
> 数字的地方都要带上这个时间戳。

扫描 `runs/*/handoffs/*/v*/validation.yaml` —— **338 份文件,941 条 verdict**。

**这是调试过程中每一次 validator 失败的账,不是当前状态。**
一条失败可能是产物缺陷、探针假象、或 validator 正确地拒绝一份 mock 材料——
本文件只记录发生过什么,归因写在各条下面,未归因的明说未归因。

跨 **34** 次运行,**46** 条失败,**11** 个 validator。
时间跨度 2026-09-04T07:48 .. 2026-09-05T07:15。

*(沿革:第一版 27 运行 / 37 条,生成时即已落后 7 条;06:57Z 重建版 33 / 45;
本版 34 / 46。)*

---

## 本文件的取样规则 —— 先说清楚,因为上一版的规则和它的实际行为不一致

**1. 每一条失败都来自被评分的阶段。** 46 条的 `environment.source` 全部是
`producer`,46 条全部带 `environment.zone`。**不采用任务 zone 里的临时目录**——
上一版有一条引自一次零 verdict 运行中手工跑的自检。

**2. `note:` 不是失败理由。** 理由只取产物 `validator_report.txt` 里的
`PROBLEM:` 行,**并且以产物为准,不以日志的标题为准**——日志曾把 `note:` 重新
标成 `PROBLEM:`(即 T49 发生在账本自己身上)。

**第一版里引了 `note:` 的是四条,不是两条**:`check_bench_result`、
`check_measurement_order`、`check_trace_coverage`、`check_no_regression`;其中前
两条还被改写上了 `PROBLEM:` 前缀。四条现在都已换成真正的 `PROBLEM:` 行。
06:57Z 那一版把这个数字写成了两条——**一份自审如果低报自己的错误率,那正是全文里
没有第二个人会去核的那个数字**,所以在这里更正,而不是默默改掉。

**3. 崩溃与拒绝没有被区分,而且本文件的方法论无法区分。** 我打开过的每一份报告,
头部都明写 `REFUSED`。而一个在写出 `verdict.json` 之前就死掉的 body,按框架行为
**根本不会留下 verdict 行**——所以一次崩溃会**缺席**于这份普查,而不是在里面被错
分类。**不要拿本文件去推崩溃率**:它没有那个分母。这个问题是开放的。

**4. 「同名取第一条」的真实含义是「第一条留下了 report 的」**,对于后期才补上
报告的 validator,这条规则会系统性地取到**较晚**的那次失败。**本版按 `at` 时间
排序取最早的一次;当最早那次没有留下 report 时,明确标注所引的不是最早一次。**

**5. 没有理由就写没有理由**,不用一条看起来合理的话替代。这里有**两个不同的数字,
都要带上定义**,不能互相替代:

- **15 条 / 46 条 = 33%** —— **没有留下任何理由的失败**总数。
- **10 条** —— 来自**两个从不写 report 的 body**(`check_deploy_serves` 7、
  `check_environment` 3)。这两条是"永远不会有理由"。
- 另 **2 条** 是 `check_deploy_kit`:它**会**写 report(全部 verdict 里 45/57),
  偏偏两次失败都没写。是"这两次没留下",不是"这个 body 不写"。
- 另 **3 条** 是 `check_workset_runs`、`check_optimization_shape`、
  `check_speedup_substantiated` 各自**最早的那一次**失败(见规则 4),它们所属的
  validator 平时也是写 report 的。

**「总是没有理由的 validator」和「没有理由的失败」是两句不同的话**,不能互相替代:
06:57Z 版只写了一个把三者混在一起的旧计数(11 条 / 24%)。分项见文末「零理由」。

**6. 不属于本文件的:** 今天三条 profiling 线的死亡(`p4_a`、`p4_b`、
`p4_m4real`)**不是 validator 失败**——`0 validation(s) dropped`,没有任何拒绝,
分配仍然持有。它们记在 `bug.record.2026-09-05.md`,原因未定。

---

## 计数

| 次数 | validator | 出现的运行 |
|---|---|---|
| 7 | `check_workset_runs` | 081016, 082227, 083225, 084313, 091607, 092903, 110626 |
| 7 | `check_speedup_substantiated` | 085348, 094614, 101432, 102832, 104255, 105342, **20260905T051226** |
| 7 | `check_deploy_serves` | 091312, 125637, **20260905T041023, 051029, 055212, 055216, 065233** |
| 5 | `check_optimization_shape` | 085348, 094614, 110647, 131949, **20260905T051226** |
| 5 | `check_trace_coverage` | **20260905T040341, 041258, 043332, 050803, 060945** |
| 4 | `check_no_regression` | 112414, 114914, 133028, 143952 |
| 3 | `check_environment` | 091312, 091607, **20260905T051226** |
| 3 | `check_workset_shape` | 091607, 092903, 110626 |
| 2 | `check_deploy_kit` | 074821, 091312 |
| 2 | `check_measurement_order` | 143952(同一次运行两条) |
| 1 | `check_bench_result` | 183810 |

未加粗的运行编号省略了 `20260904T` 前缀。**加粗的是 2026-09-05 的运行**——
`check_trace_coverage` 的 5 条**全部**发生在今天,`check_deploy_serves` 的 7 条
里有 5 条在今天。**这两个是当前仍在产生失败的。**

---

## 每个 validator 的代表性理由

摘自该次失败自己 zone 里的 `validator_report.txt`,只取 `PROBLEM:` 行。

**`check_workset_runs`** — 7 条,6 条留下 report。**所引非最早一次**
(最早 08:15:51 无 report,引 08:28:09)。
> PROBLEM: sampler_vocab_softmax/B8_V151936: the performance entrypoint exited 1:
> files through the zone, which needs the run root to be on a filesystem both
> hosts mount…

**注意**:这段文字里的 `-v /home:/home denied` 一族**是
`measure_in_container.sh` 拒绝分支里的「已知不可用形式」目录行,不是当次发生的
守护进程拒绝**。判断这 7 条的真实原因时不要把它当成事件。

**`check_speedup_substantiated`** — 7 条,6 条留下 report。**所引非最早一次**
(最早 08:59:47 无 report,引 09:52:10)。
> PROBLEM: ABORT — this handoff optimised sampler_vocab_softmax at dtype None;
> the workset's ground truth says 'float32'. A speedup at a different precision
> is a different question

**`check_deploy_serves`** — 7 条,**0 条留下 report**。
> **理由不存在,而且永远不会有。** 见下文「零理由」一节:这个 body 在
> **全部 55 次 verdict**(通过的也算)里一次都没写过 report。

**`check_optimization_shape`** — 5 条,4 条留下 report。**所引非最早一次**
(最早 08:59:45 无 report,引 09:52:08)。
> PROBLEM: kernel_optimization: 1 problem(s)

**`check_trace_coverage`** — 5 条,5 条全部留下 report。所引即最早一次。
> PROBLEM: expected 4 rank(s), the manifest lists 2

**`check_no_regression`** — 4 条,4 条全部留下 report。所引即最早一次。
> PROBLEM: the report was decided against max_throughput_regression=35% and this
> validator's bar is 5%. **A producer that chooses its own threshold can pass
> anything.** …
> PROBLEM: the report was decided against max_latency_regression=30% and this
> validator's bar is 10%. …
> PROBLEM: the patch regressed: output_token_throughput_tps (avg),
> request_throughput_rps (avg), ttft_ms (p90), request_latency_ms (p90) …
> PROBLEM: this run cannot judge the patch — 6 metric(s) have a noise floor above
> their bar …

> **这四条不是缺陷,是本包刻意保留的那一条 strong 拒绝——本账里最需要交叉引用的
> 一行。**
>
> 同一份报告里还写着 `recomputed: REJECTED (10 reason(s)); report states
> REJECTED`:validator 不采信报告的 `verdict` 字段,从原始数字**独立重算**,得到
> REJECTED,**并且与报告的结论一致**。这就是 `RUN-PLAN.md`「mock 运行退出码是 5,
> 而 5 是正确输出」那一节说的同一个事件。
>
> **而 m5 的 `1cda806` 确立了另一半:`--var mock_report=accepted` 关掉的正是这条
> 拒绝。** 证据是逐字对上的——上面第一条 PROBLEM 说 *"a producer that chooses its
> own threshold can pass anything"*,而 `1cda806` 的 commit message 正是引用这句话
> 来说明该开关移除了什么:它用测量出的对照数字和 validator 自己的 5%/10% 替换掉封存
> 报告里 35%/30% 的 PRODUCER bar。
>
> 所以:**本账里最难读的一行,和唯一能让它消失的那个开关,指向同一个事件。**
> `1cda806` 还记了一句必须一起读的:在 accepted 模式下,整条链**不再**行使本包唯一
> 那条廉价的端到端 strong 拒绝(`mock_m5.sh:140`),所以那种情况下的「绿」是一个更
> 窄的主张。
>
> 只引第一条 PROBLEM 而不写上面这些,这一行读起来就只是一次普通失败,没有人会把它
> 和那个开关联系起来。

**`check_environment`** — 3 条,**0 条留下 report**。
> **理由不存在,而且永远不会有。** 同 `check_deploy_serves`:这个 body 在
> **全部 338 次 verdict** 里一次都没写过 report。见文末「零理由」。

**`check_workset_shape`** — 3 条,3 条全部留下 report。所引即最早一次。
> PROBLEM: items/codes/workset.yaml does not load: [Errno 2] No such file or
> directory: '…/runs/20260904T091607-e5f80f/zones/…'

**`check_deploy_kit`** — 2 条,**0 条留下 report**。
> **理由不存在。** 但与上面两个不同:这个 body **会**写 report(全部 verdict 里
> 45/57),只是这两次失败没留下。所以是"这两次丢了",不是"从不写"。
>
> *(第一版此条引了一句 `PROBLEM: packup/REPRODUCE.md: unfilled placeholder
> '<user>'`,**来源是错的**——那句唯一的出处是一次**零 verdict 运行**的 task zone
> 临时目录里手工跑的自检,且那个运行不是本行列出的两个之一。已删除且不另找替换。)*

**`check_measurement_order`** — 2 条,2 条全部留下 report。所引即最早一次。
> PROBLEM: stock: environment.yaml says node='crsuse2-m2m-217' and the arm's own
> evidence says 'crsuse2-m2m-276'. The record describes the machine…

*(上一版此条引的是 `note: stock: serve -> smoke -> …`,那是一条 note,不是理由。)*

**`check_bench_result`** — 1 条,留下 report。
> PROBLEM: bench: the engine captured decode graphs up to batch 16
> (`--cuda-graph-max-bs` → 16) and the load achieved decode concurrency 25.42

*(上一版此条引的是 `PROBLEM: (note) 721 request record(s), 0 errored (0.0%)`
——一条被标成 `PROBLEM:` 的 note。)*

---

## 零理由 —— 15 条 / 46 条 = 33%


**没有留下理由的失败共 15 条,占 46 条的 33%**,分成三类(规则 5):

| 类别 | 条数 | 含义 |
|---|---|---|
| 两个**从不写 report** 的 body | **10** | 永远不会有理由 |
| `check_deploy_kit` 的两次失败 | **2** | 这个 body 会写,这两次没留下 |
| 三个 validator 各自**最早**的一次失败 | **3** | 报告能力是后来才补上的(规则 4) |

### 那两个从不写 report 的 body,以及为什么"永远"不是修辞

关键不是"这几次运行把文件弄丢了",而是这几个 body **从来没写过**。区别在一次运行上
根本看不出来,分数才是证据——**分母是全部 verdict,通过的也算**:

| validator | 失败条数 | 全部 verdict 中留下 report 的比例 |
|---|---|---|
| `check_deploy_serves` | 7 | **0 / 55** |
| `check_environment` | 3 | **0 / 338** |
| `check_deploy_kit` | 2 | 45 / 57 |

前两个是 **0/55** 和 **0/338**——一次都没有。`check_deploy_kit` 是对照组,它**会**
写 report(57 次里 45 次),偏偏两次失败都没写;所以它属于"这两次没留下",而不是
"这个 body 不写"。

而 `verdict.json` 里只有 `{"<handoff_id>": false}`,**没有任何 reason 字段**(11 条
首次失败逐个查过)。**所以这 12 条在磁盘上没有任何理由,而且以后也不会有。** 它们是
真实发生的失败,永久无法解释。

这不是记录的疏漏,是框架的缺口:validator 把诊断写到 stdout,而 stdout 不被保存
(`bug.record.2026-09-05.md` 第 2 条)。写在这里是为了让下一个人知道分母,而不是
以为这三个 validator 失败得比较安静。

### 归因上的保留意见,必须一起读

checkpoint 从另一个方向独立走到了同样这几行,并称之为"披着 validator 名字的框架
bug"。两条路径的结论一致——**但两次计数很可能共用同一个来源,即 `verdict.json` 不带
reason 字段这一事实**。所以两者一致的是**机制**;至于该不该把它叫做框架 bug,
是一个判断,**两次计数都没有确立它**,谁也不应该被读成已经确立了它。
