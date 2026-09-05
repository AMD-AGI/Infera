# validator 失败记录 — 全量收割

生成于 **2026-09-05T04:59:44Z**，扫描 `runs/*/handoffs/*/v*/validation.yaml`。
2026-09-05 晚经过一次逐条核对（见文末"核对记录"），代表性理由一栏被改写过——
**改写之前有 5 条是错的或引错了行**，原文没有保留，因为它们的内容本身就是错的。

**这是调试过程中每一次 validator 失败的账，不是当前状态。**
一条失败可能是产物缺陷、探针假象、环境/管道问题，或 validator 正确地拒绝一份 mock
材料——本文件只记录发生过什么，归因写在各条下面，未归因的明说未归因。

---

## 读这份文件之前：三条规则

**1. "代表性理由"的选取规则不是"第一条"，而是"第一条留下了报告的"。**
文件原本写的是*"同名只取第一条"*。实际做不到：一条失败的理由**只存在于
`validator_report.txt` 里**，而 `verdict.json` 只有 `{"<handoff_id>": false}`，
没有任何 reason 字段。所以选取规则实际是"该 validator 第一条**留下了报告**的失败"。

**这个偏差是系统性的，不是随机的。** 各个 validator 的 body 是在不同时间才被加上
写报告的能力的，所以对**较晚才被插桩的那些 validator**，代表性理由必然取自**较晚的
那一次失败**，而不是本行"出现的运行"列里的第一个运行。已确认的三例：

| validator | 本行第一个运行 | 理由实际取自 |
|---|---|---|
| `check_workset_runs` | `20260904T081016`（无报告） | `20260904T082227` |
| `check_optimization_shape` | `20260904T085348`（无报告） | `20260904T094614` |
| `check_speedup_substantiated` | `20260904T085348`（无报告） | `20260904T094614` |

**2. "validator 失败"不等于"产物有问题"。** 至少两条是**管道/环境问题，产物根本没
被读到**：`check_workset_runs` 的 demo-root 拒绝、`check_workset_shape` 的
`workset.yaml` 找不到，两者都是 zone / run root 路径问题。

**3. 崩溃与拒绝没有在这份账里被区分，而且这份账**无法**区分。**
我打开过的每一份报告，头部都明写 `REFUSED`。而一个在写出 `verdict.json` 之前就死掉的
body，按框架的行为**根本不会留下 verdict 行**——所以一次崩溃会**缺席**于这份普查，
而不是在里面被错分类。**这个问题在本文件里是开放的，且本文件的方法论无法回答它。**

---

## 普查（截至 2026-09-05T04:59:44Z）

跨 **27** 次运行，**37** 条失败，**11** 个 validator。
（2026-09-05 晚独立重算过一次，逐个 validator 的次数与运行列表完全复现。）

| 次数 | validator | 出现的运行 |
|---|---|---|
| 7 | `check_workset_runs` | 20260904T081016, 20260904T082227, 20260904T083225, 20260904T084313 …共 7 次 |
| 6 | `check_speedup_substantiated` | 20260904T085348, 20260904T094614, 20260904T101432, 20260904T102832 …共 6 次 |
| 4 | `check_optimization_shape` | 20260904T085348, 20260904T094614, 20260904T110647, 20260904T131949 |
| 4 | `check_no_regression` | 20260904T112414, 20260904T114914, 20260904T133028, 20260904T143952 |
| 3 | `check_deploy_serves` | 20260904T091312, 20260904T125637, 20260905T041023 |
| 3 | `check_workset_shape` | 20260904T091607, 20260904T092903, 20260904T110626 |
| 3 | `check_trace_coverage` | 20260905T040341, 20260905T041258, 20260905T043332 |
| 2 | `check_deploy_kit` | 20260904T074821, 20260904T091312 |
| 2 | `check_environment` | 20260904T091312, 20260904T091607 |
| 2 | `check_measurement_order` | 20260904T143952（同一次运行里两条） |
| 1 | `check_bench_result` | 20260904T183810 |

### 该时间点之后新增的（补记，2026-09-05 晚）

再 **7** 条，跨 **5** 次运行，未并入上表。

| 运行 | validator | 理由 |
|---|---|---|
| 20260905T050803 | `check_trace_coverage` | `expected 4 rank(s), the manifest lists 2` |
| 20260905T051029 | `check_deploy_serves` | 无记录（见"永远无法解释的五条"） |
| 20260905T051226 | `check_environment` | 无记录（同上） |
| 20260905T051226 | `check_optimization_shape` | `missing results/kernel_optimization.json; there is nothing structured to consume` |
| 20260905T051226 | `check_speedup_substantiated` | `missing results/kernel_optimization.json` |
| 20260905T055212 | `check_deploy_serves` | 无记录 |
| 20260905T055216 | `check_deploy_serves` | 无记录 |

---

## 每个 validator 的一条代表性理由

摘自 run 树里的 `validator_report.txt`，取该 validator 第一条留下了报告的失败
（见上文规则 1）。**报告里的 `note:` 行不是失败理由**——下面只引 `PROBLEM:` 行。

**`check_bench_result`**（20260904T183810）
PROBLEM: bench: the engine captured decode graphs up to batch 16 (`--cuda-graph-max-bs -> 16`) and the load achieved decode concurrency **25.42** — so decode exceeded the captured graph on essentially every step and the engine fell back to **eager decode**. Measured on this cluster: a 4.6x difference in decode speed from this cause alone, with an identical image and node.

> **本条曾被引错，而且错的方式正是 T49。** 原文引的是
> `PROBLEM: (note) 721 request record(s), 0 errored (0.0%)`——那一行在产物里是
> **`note:`**，内容是"零错误"，也就是一条**通过**的观察；本文件把它的 `note:`
> 前缀改写成了 `PROBLEM:`，于是一条报告"零错误"的记录被当成了失败的原因。
> **记录缺陷的工具，犯了它所记录的那个缺陷。**

**`check_deploy_kit`**（20260904T074821, 20260904T091312）
**无记录的理由。** 这两条失败都没有留下 `validator_report.txt`。

> 原文此处引了一句 `PROBLEM: packup/REPRODUCE.md: unfilled placeholder '<user>'`，
> **来源是错的**：那句话唯一的出处是
> `runs/20260904T173415-35b7a3/zones/.../tmp/tmp.O4qB4jcMsz/validator_report.txt`
> ——一个 **task zone 里的临时目录**，属于一次**没有记录任何 verdict** 的运行，
> 也就是一次手工自检，而不是被评判的 phase；且 `20260904T173415` 不是本行列出的
> 两个运行中的任何一个。已删除，**不另找一句替换**：这两条失败没有理由留下来，
> 就该这么写。（`check_deploy_kit` 本身是会写报告的——全部 51 次 verdict 里有
> 39 次留下了报告——所以这不是"这个 validator 从不写报告"。）

**`check_deploy_serves`**
**无记录的理由，且永远不会有。** 见"永远无法解释的五条"。

**`check_environment`**
**无记录的理由，且永远不会有。** 见"永远无法解释的五条"。

**`check_measurement_order`**（20260904T143952）
PROBLEM: stock: `environment.yaml` says `node='crsuse2-m2m-217'` and the arm's own evidence says `'crsuse2-m2m-276'`. …
PROBLEM: stock: `environment.yaml` says `slurm_jobid='109491'` and the arm's own evidence says `'101053'`. …
（patched 臂两条同样。）

> 原文引的是 `note: stock: serve -> smoke -> needle -> …` 那几行——那是两臂的步骤
> 顺序，而且它们是**一致的**，即这条检查通过的那一半。真正的拒绝理由是记录与证据
> 对不上机器。

**`check_no_regression`**（20260904T112414）
PROBLEM: the report was decided against `max_throughput_regression=35%` and this validator's bar is **5%**. **A producer that chooses its own threshold can pass anything.** …
PROBLEM: the report was decided against `max_latency_regression=30%` and this validator's bar is **10%**. …
PROBLEM: the patch regressed: `output_token_throughput_tps` (avg), `request_throughput_rps` (avg), `ttft_ms` (p90), `request_latency_ms` (p90) …
PROBLEM: this run cannot judge the patch — 6 metric(s) have a noise floor above their bar …

> **这四条不是缺陷，是本包刻意保留的那一条 strong 拒绝。**
>
> 原文引的是 `note: ! kernel reconciliation not computed…`，读起来像"有个东西没算
> 出来"。实际上同一份报告里还写着
> `recomputed: REJECTED (10 reason(s)); report states REJECTED`——validator 从原始
> 数字独立重算，得到 REJECTED，**与报告的结论一致**。这正是
> `RUN-PLAN.md` 里"mock 运行退出码是 5 且 5 是正确输出"那一节说的同一件事。
>
> **而 m5 的 `1cda806` 确立了另一半：`--var mock_report=accepted` 会关掉的正是
> 这条拒绝。** 该 commit 的原话——`mock_report=accepted` 用测量出的对照数字和
> validator 自己的 5%/10% 替换掉封存报告里 35%/30% 的 PRODUCER bar，*"而这恰恰就是
> 那条封存拒绝所针对的东西——'一个自己挑阈值的 producer 什么都能通过'"*。把上面第
> 一条 PROBLEM 逐字对照一下就能看到是同一句话。
>
> 所以：**本账里最难读的一行，和唯一能让它消失的那个开关，指向的是同一个事件。**
> 1cda806 还记了一句必须一起读的：在 accepted 模式下，整条链**不再**行使本包唯一
> 那条廉价的端到端 strong 拒绝（`mock_m5.sh:140`），所以那种情况下的"绿"是一个更
> 窄的主张。

**`check_optimization_shape`**（20260904T094614）
PROBLEM: kernel_optimization: 1 problem(s)
PROBLEM:   `$.apply.integration_point.entry_function: '' is too short`

**`check_speedup_substantiated`**（20260904T094614）
PROBLEM: ABORT — this handoff optimised `sampler_vocab_softmax` at dtype `None`; the workset's ground truth says `'float32'`. A speedup at a different precision is a different question.

**`check_trace_coverage`**（20260905T040341）
PROBLEM: `expected 4 rank(s), the manifest lists 2`

> 本条也曾被引错，同 `check_bench_result` 的机制：原文引的是
> `PROBLEM: (note) re-parsed … manifest agrees`——产物里那是 `note:`，而且
> "manifest agrees" 说的是**对上了**，是通过的那一半，`note:` 前缀同样被改写成了
> `PROBLEM:`。

**`check_workset_runs`**（20260904T082227 —— 不是本行第一个运行）
PROBLEM: `sampler_vocab_softmax/B8_V151936`: the performance entrypoint exited 1: files through the zone, which needs the run root to be on a filesystem | both hosts mount. Point `--demo-root` at a shared path (`$HOME` here is NFS | and works).

> 引文逐字属实（包括那两个 `|`，原文就有）。**这是环境/管道问题，不是产物缺陷**：
> workset 根本没被跑起来。

**`check_workset_shape`**（20260904T091607）
PROBLEM: `items/codes/workset.yaml` does not load: `[Errno 2] No such file or directory: '/home/yihou/agent_sys_runroot/runs/20260904T091607-e5f80f/zones/task.…'`

> 同上，**产物没被读到**，不是产物有问题。

---

## 永远无法解释的五条

`check_deploy_serves`（3 条）和 `check_environment`（2 条）。

原文写的是*"该 validator 未留下 `validator_report.txt`"*，事实比这句更强。跨**所有**
已记录的 verdict，通过的和失败的一起算：

```
check_deploy_serves    0/ 51 次 verdict 留下了 validator_report.txt
check_environment      0/319 次 verdict 留下了 validator_report.txt
```

**不是这几次运行把文件弄丢了——这两个 body 从来没有写过报告。** 而
`verdict.json` 里只有 `{"<handoff_id>": false}`，没有任何 reason 字段（11 条首次
失败逐个查过）。

**所以这五条失败在磁盘上没有任何理由，而且以后也不会有。** 它们是真实发生的失败，
但永久无法解释。

**归因上的保留意见，必须一起读：** checkpoint 从另一个方向独立走到了同样这五行，
并称之为"披着 validator 名字的框架 bug"。两条路径的结论一致——**但两次计数很可能
共用同一个来源，即 `verdict.json` 不带 reason 字段这一事实**。所以两者一致的是
**机制**；至于该不该把它叫做框架 bug，是一个判断，**两次计数都没有确立它**。

---

## 核对记录

2026-09-05 晚，逐条对着 run 树核对了一次。方法：从
`runs/*/handoffs/*/v*/validation.yaml` 独立重建普查，按 `at < 04:59:44Z` 过滤；
再把每条代表性理由的引文回grep到 275 个 `validator_report.txt` 里，确定它到底出自
哪一次 verdict、那次 verdict 是通过还是失败。

结论：

- **普查表完全属实**，37 / 27 / 11 与逐 validator 的次数、运行列表全部复现。
- **1 条来源错误**：`check_deploy_kit`（手工自检的输出，冒充两条失败的理由）。
- **4 条引错了行**：`check_bench_result`、`check_trace_coverage`、
  `check_measurement_order`、`check_no_regression`，引的都是 `note:`；其中前两条
  还被改写成了 `PROBLEM:` 前缀。
- **2 条属实但说轻了**：`check_deploy_serves` / `check_environment`。
- **0 条**是"把崩溃当成 validator 缺陷"——且见上文规则 3，这份账无法回答这个问题。
- 27 个被点名的运行**全部存在**，没有一条因为运行消失而无法核实。
