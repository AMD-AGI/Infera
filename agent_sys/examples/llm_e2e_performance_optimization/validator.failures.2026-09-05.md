# validator 失败记录 — 全量收割

重新生成于 2026-09-05T06:57Z,**独立重建**,不是在旧文件上追加。
扫描 `runs/*/handoffs/*/v*/validation.yaml` —— **327 份文件,912 条 verdict**。

**这是调试过程中每一次 validator 失败的账,不是当前状态。**
一条失败可能是产物缺陷、探针假象、或 validator 正确地拒绝一份 mock 材料——
本文件只记录发生过什么,归因写在各条下面,未归因的明说未归因。

跨 **33** 次运行,**45** 条失败,**11** 个 validator。
时间跨度 2026-09-04T07:48 .. 2026-09-05T06:18。

*(上一版为 27 次运行 / 37 条 / 11 个 validator,生成时即已落后 7 条。)*

---

## 本文件的取样规则 —— 先说清楚,因为上一版的规则和它的实际行为不一致

**1. 每一条失败都来自被评分的阶段。** 45 条的 `environment.source` 全部是
`producer`,45 条全部带 `environment.zone`。**不采用任务 zone 里的临时目录**——
上一版有一条引自一次零 verdict 运行中手工跑的自检。

**2. `note:` 不是失败理由。** 理由只取产物 `validator_report.txt` 里的
`PROBLEM:` 行,**并且以产物为准,不以日志的标题为准**——日志曾把 `note:` 重新
标成 `PROBLEM:`(即 T49 发生在账本自己身上)。**上一版 `check_bench_result` 和
`check_measurement_order` 两条引的都是 `note:`,本版已换成真正的 `PROBLEM:` 行。**

**3. 「同名取第一条」的真实含义是「第一条留下了 report 的」**,对于后期才补上
报告的 validator,这条规则会系统性地取到**较晚**的那次失败。**本版按 `at` 时间
排序取最早的一次;当最早那次没有留下 report 时,明确标注所引的不是最早一次。**

**4. 没有理由就写没有理由**,不用一条看起来合理的话替代。
**45 条里有 11 条属于此类**,分布在 3 个 validator 上。

**5. 不属于本文件的:** 今天三条 profiling 线的死亡(`p4_a`、`p4_b`、
`p4_m4real`)**不是 validator 失败**——`0 validation(s) dropped`,没有任何拒绝,
分配仍然持有。它们记在 `bug.record.2026-09-05.md`,原因未定。

---

## 计数

| 次数 | validator | 出现的运行 |
|---|---|---|
| 7 | `check_workset_runs` | 081016, 082227, 083225, 084313, 091607, 092903, 110626 |
| 7 | `check_speedup_substantiated` | 085348, 094614, 101432, 102832, 104255, 105342, **20260905T051226** |
| 6 | `check_deploy_serves` | 091312, 125637, **20260905T041023, 051029, 055212, 055216** |
| 5 | `check_optimization_shape` | 085348, 094614, 110647, 131949, **20260905T051226** |
| 5 | `check_trace_coverage` | **20260905T040341, 041258, 043332, 050803, 060945** |
| 4 | `check_no_regression` | 112414, 114914, 133028, 143952 |
| 3 | `check_environment` | 091312, 091607, **20260905T051226** |
| 3 | `check_workset_shape` | 091607, 092903, 110626 |
| 2 | `check_deploy_kit` | 074821, 091312 |
| 2 | `check_measurement_order` | 143952(同一次运行两条) |
| 1 | `check_bench_result` | 183810 |

未加粗的运行编号省略了 `20260904T` 前缀。**加粗的是 2026-09-05 的运行**——
`check_trace_coverage` 的 5 条**全部**发生在今天,`check_deploy_serves` 的 6 条
里有 4 条在今天。**这两个是当前仍在产生失败的。**

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

**`check_deploy_serves`** — 6 条,**0 条留下 report**。
> **理由不存在。** 六次失败没有任何一次留下 `validator_report.txt`,所以无法说明
> 它们为什么失败。这与 `bug.record.2026-09-05.md` 第 2 条(validator 的 stdout
> 无处可寻)是同一道缝。

**`check_optimization_shape`** — 5 条,4 条留下 report。**所引非最早一次**
(最早 08:59:45 无 report,引 09:52:08)。
> PROBLEM: kernel_optimization: 1 problem(s)

**`check_trace_coverage`** — 5 条,5 条全部留下 report。所引即最早一次。
> PROBLEM: expected 4 rank(s), the manifest lists 2

**`check_no_regression`** — 4 条,4 条全部留下 report。所引即最早一次。
> PROBLEM: the report was decided against max_throughput_regression=35% and this
> validator's bar is 5%. A producer that chooses its own threshold can pass
> anything…

**`check_environment`** — 3 条,**0 条留下 report**。
> **理由不存在。** 同 `check_deploy_serves`。

**`check_workset_shape`** — 3 条,3 条全部留下 report。所引即最早一次。
> PROBLEM: items/codes/workset.yaml does not load: [Errno 2] No such file or
> directory: '…/runs/20260904T091607-e5f80f/zones/…'

**`check_deploy_kit`** — 2 条,**0 条留下 report**。
> **理由不存在。**

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

## 三个 validator,零理由

`check_deploy_serves`(6)、`check_environment`(3)、`check_deploy_kit`(2)
—— **共 11 条失败,占 45 条的 24%,一条理由都没有留下。**

这不是记录的疏漏,是框架的缺口:validator 把诊断写到 stdout,而 stdout 不被保存
(`bug.record.2026-09-05.md` 第 2 条)。**这 11 条今天已经不可复原**,写在这里是
为了让下一个人知道分母,而不是以为这三个 validator 失败得比较安静。
