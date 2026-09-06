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

### 2.2 收敛运行:第一次在真实验证下走 mock 数据

**run `20260905T113109-2d64f3`,21 个 validator 里 20 个是活的**(只有
`check_deploy_serves` 被换掉——它要做一次真实 bring-up 和 180 秒负载)。我第一手
从 `validation.yaml` 数出来的:

| validator | 调用 | 拒绝 |
|---|---:|---:|
| `check_environment` | 9 | 0 |
| `check_command_parses` | 4 | 0 |
| `check_bench_result` | 2 | 0 |
| `check_deploy_kit` · `check_identity_resolved` · `check_kernel_table` · `check_profiling_evidence` · `check_trace_coverage` · `check_worklist_shape` | 各 1 | 0 |
| `check_workset_shape` | 1 | **1** |
| `check_workset_runs` | 1 | **1** |

**共 23 次 verdict、11 个 validator、2 次拒绝。** 两次都在 `build_workset`,都正确:
封存 workset 没有 evidence 块,而 M4.3.5 要求 m4 严格从这件产物取 ground truth,
所以一件没被测量过的 workset **没有东西可供它取**。和 environment 记录同一族——
**2026-09-02 的语料早于它现在要满足的契约**。

**阶段 1 和阶段 2 现在在真实验证下、跑 mock 数据、全部通过;阶段 3 是语料用尽的地方。**

### 2.3 这一轮的三次拒绝,没有一次是 validator 的缺陷

| 拒绝 | 归因 |
|---|---|
| `check_trace_coverage`(run `112313-17c51a`) | **发射参数错误**——封存 trace 是 2 rank,启动时没带 `--var expect_ranks=2`。带上之后它通过,而且是在**重新解析了一个 340 万事件的 gzip** 之后通过的 |
| `check_workset_shape` | 语料早于契约 |
| `check_workset_runs` | 语料早于契约 |

**一次是操作者的,两次是语料的,零次是 validator 的。**

### 2.4 关键的一点:这次通过的那些,恰好就是被负控制证明过的那些

把 2.2 的通过名单和 2.1 的负控制名单并排看:

```
2.1 被负控制证明会判分:  command_parses  kernel_table  identity_resolved
                          profiling_evidence  worklist_shape
2.2 本次通过且零拒绝的:   command_parses  kernel_table  identity_resolved
                          profiling_evidence  worklist_shape  (+ 三个本来就拒绝过的)
```

**完全重合。** 这把它们的 PASS 从「不知道意味着什么」变成了**关于产物的信息**:
我们独立地知道,给它们坏输入时它们会拒绝,所以这次它们没拒绝,说的是**产物是好的**,
不是「这个检查不判分」。

**而剩下五个从未拒绝过的——`check_acceptance`、`check_bench_report`、
`check_overlay_applies`、`check_patch_live`、`check_packup_shape`——全部是 m5 的**,
这和 m5 从未真实运行过是同一件事的两种说法。它们仍然是本包证据最薄的一角。

### 2.5 `check_environment` 的九次通过买到了什么——以及买不到什么

**这是本包调用最多、接触面最广的 validator**(全部 15 个 kind 都挂着它,375 次调用)。
2.2 里它在收敛运行中 **9 次调用、0 次拒绝**。那九次确立了什么,可以精确地说。

**我第一手数的,同一个运行 `20260905T113109-2d64f3`:**

```
9 record(s):  node='mock'   node_ip='127.0.0.1'   gpu_arch='gfx950'
```

**九条 environment 记录,`fixed.node` 是字符串 `mock`,`node_ip` 是回环地址,而
`gpu_arch` 声称 `gfx950`——运行发生在一台没有 GPU 的登录节点上。** 而
`check_environment` 九次全部通过,`strong`。这九次通过和那九条记录是同一批。

**所以它确立的是:记录存在、通过 schema、并且彼此一致。**
`compare_fixed_across_inputs` 比的是几个 handoff **互相之间**是否一致——**不是它们
和机器之间**是否一致。**一份记录可以完整、合法、内部自洽,并且说的是另一台机器。**

#### 撤回:本节原本还有第二个实例,那个实例是假的

**原文写着**:一份在 287 上产出的 `deploy_kit`,记录却写着
`node: crsuse2-m2m-217` / `yihou_p5_217_p5main`,而 `check_environment` 以 `strong`
放过了三次;并由此推出这个守卫「被反了过来」。**那是错的,现予撤回。** m1 测量、
m5 同意并撤回,我自己复核了产出方:

```
run 20260905T084736-f0a236   deploy_and_prove  agent=e2e_deployer  succeeded
  kit record:  node=crsuse2-m2m-217  ip=10.245.155.122
               container=yihou_p5_217_p5main   replayed_from=None
```

**那份 kit 就是在 217 上产出的,记录说 217 是真话。** `p5_217` 是那次运行自己的
`--var container=`,不是某个 fallback 盖过了传入值。m5 是**在 287 上重放**它,不是
在 287 上产出它——他们那份记录带着
`replayed_from: 20260905T084736-f0a236`、`container: replayed-from-…-NOT-RUNNING`、
`endpoint: …invalid:0`。**`--var node=287` 说的是「谁在重放」,记录说的是「谁产出
了这件产物」——这是两个问题,而记录回答的是对的那个。**

而且这正是 `replay_root.py` 写明的决定:*"`fixed.node` is deliberately NOT
rewritten … letting `_agree_or_die` and `compare_fixed_across_inputs` refuse is the
correct outcome"*。他们用 `--allow-cross-node` 绕过的那次拒绝,**是机制在正常工作**。

**所以「守卫被反过来」这个推论一并撤回。** 它建立在那个假实例上,没有独立证据。

#### 撤回之后剩下什么 —— 而剩下的本来就是全部

**§2.5 开头那九条 `node: mock` 记录是我第一手测的,它不需要第二个例子。**
九条声称 `gfx950` 的记录来自一台没有 GPU 的登录节点,九次 `strong` 通过。

**而且结构性的限制被第二条路径独立确认了**(m1):把 `check_environment` 的 body
grep 一遍找 subprocess / 远程调用,**结果是零**。**它按构造就不接触任何节点。**
所以「记录存在、合法、与兄弟一致」正是九次通过买到的东西,而「与这台机器相符」
不在其中——这一点不依赖任何一个错误的例子。

**开放,而不是舒服地关掉**(m1 的措辞):这份 kit 是真的,但**没有任何东西表明一份
写错节点的 kit 不可能存在**——上面那个盲区意味着它可以存在,而且不会被抓住。
**这才是真正的发现,它不需要一个假例子来支撑。**

**这和 `SKIP-AHEAD.md` §6.1 是同一个机制的两面。** 那边是**继承**:每个下游 handoff
用 `env_render --inherit <被 replay 的 kit>` 渲染,四个被比对的字段全是从同一份记录拷
来的,于是彼此一致、比对通过、而运行根本不在那台机器上。这边是**铸造**:记录一开始就
写错。两条路径,同一个结果——**一份自洽且不真的记录**,而比较看不见所有参与方共有的
故障(CONTRACT §4.6)。

**不在本文件里修。** 产生这份记录的代码是 m1 的,leader 已经带着「`p5_217` 是从哪来
的」这个问题路由过去了。这里只说清楚九次通过买到了什么。

### 2.6 证据最薄的五个,和从未真实运行的那个阶段,是同一件事

2.1 之后仍未被负控制驱动过的五个——`check_acceptance`、`check_bench_report`、
`check_overlay_applies`、`check_patch_live`、`check_packup_shape`——**全部属于 m5**。
这不是两个缺口,是一个:**本包证据最薄的一角,和它唯一从未真实运行过的阶段,是同一个
事实的两种说法。**

### 2.7 一次 **17/17** 的 `-noval` 完成确立了什么

**run `20260905T121310-bf8226`,2026-09-05 13:31,78 分钟,五个阶段一次 `agent-sys
run` 走完。我第一手复核:17 个 task 全部 `succeeded`,含 `m5_integration`、`packup`
和 `main`;15 个 kind、21 个封存版本、495 个文件、19 份 environment 记录。**

```
真实 stage 1 (deploy_and_prove)        35m34s
语料中段 (m2+m3+m4)                     2m23s
真实 stage 5 (integrate_and_verify)    39m29s
```

#### 一、它确立了一个结构性事实,而这是任何部分运行都给不了的

**不存在走不通的阶段,stage 1 到 stage 5 之间不存在接线故障。** 今天死掉的每一条链
都死在**分歧**上(语料早于契约、发射参数错、body 失败被丢弃),没有一条死在「图接不
上」。**这一点从此是测出来的,不是论证出来的。**

#### 二、文件计数买到了什么 —— 分开算才有意义,合起来算什么都不是

leader 问 495 个文件和 19 份记录能不能收紧「静默失败」这条。**我去测了。**
判据来自我自己记录的那个失败模式:`mock.sh` 先拷贝、body 再失败、任务仍记
`succeeded`——**在那种模式下,handoff 里应当只有语料字节。**

```
全部 handoffs/ 下 495 个文件:  165 与语料逐字节相同 (33%)   330 不是 (67%)
```

逐 kind 分开:

| kind | 文件 | 语料 | 产出 |
|---|---:|---:|---:|
| `deploy_kit` | 37 | 2 | **35** |
| `patched.measurement` | 71 | 3 | **68** |
| `stock.measurement` | 71 | 6 | **65** |
| `e2e_packup` | 86 | 21 | **65** |
| `patch_overlay` | 9 | 1 | 8 |
| `operator_workset` | 41 | 26 | 15 |
| `kernel_optimization` | 33 | 19 | 14 |
| `profiling_evidence` | 45 | 34 | 11 |
| `profiling_mode_off.bench_result` | 17 | 15 | **2** |
| `profiling_mode_on.bench_result` | 17 | 15 | **2** |
| `profiling_mode_on.profile_result` | 16 | 14 | **2** |

**十五个 kind 没有一个是 100% 语料。** 但这条判据**强弱极不均匀,而不均匀的方式恰好
可以预测**:

- **真实跑过的两个阶段(m1、m5)产出占压倒多数**——35/37、68/71、65/71。这里计数是
  真证据。
- **语料中段只有 2–4 个产出文件**,而那几个**完全可以由 `mock.sh` 自己的引号修复
  (MOCK-MAP J)加上 `env_render` 写的 `environment.yaml` 解释掉**——也就是说,
  **它们正是「拷贝 → 修复 → 渲染 → 然后失败」会留下的东西。** 在这里计数买不到任何
  东西。

**所以结论是:计数只在本来就跑了真实工作的地方证明真实工作。** 它不能把
`-noval` 的绿升级成正确性,leader 原本担心的「a lot of structure 不是论证」是对的
——但它现在**是一个可算的、逐 kind 的判据**,而不是一个印象。

#### 三、这次运行里唯一一个非 `check_nothing` 的事实,是一次拒绝

**而这是全场最强的证据,强过任何计数。** `integrate_and_verify` **自己算出**
`verdict.accepted: false`:

```
stock_vs_m2   inter_token_latency (avg)   m2 15.6456 ms   stock 11.2952 ms
              rel_delta -0.278058         within_tolerance: false
comparison    3 轮/臂,每臂 pooled n=720,带 median 与 spread
```

**这不是一个 validator 的判决——`check_no_regression` 在这次运行里是
`check_nothing`。这是生产者自己的计算。** 而它需要:读到 m2 的 bench、测出 stock 臂、
把两者比较。**一个「拷贝完就失败」的 body 产不出这个。**

拒绝的理由还指名了下一步该查什么,同时明说自己不做那个判断:

> *"the two stages measured something different… Which one is decidable and this
> body does not decide it: compare `fixed.image_id` and `fixed.node` in the two
> `environment.yaml` records."*

**一个来自另一台机器的语料中段,对上一个真实的 stage 5,本来就应该不一致。**
所以这次运行里那一个实质判断,是**一次正确的拒绝**——那比一个绿更强。

#### 四、由此得到的一般判据

**在 `check_nothing` 之下,证据既不是 verdict 也不是文件数,而是「生产者算出来的、
其值依赖于它必须读进来的输入」的内容。** 一个数字如果只能由「真的读了 A 又读了 B 再
比较」才能得到,它就不能被静默失败伪造;一个文件如果只是被拷进来的,它什么都不证明。

**用这条去读上面三点:** 结构性可达(一)是真的;计数(二)只在真实阶段有效;
而唯一不可伪造的那一条(三)恰好是本次运行**唯一一个说「不」的东西**。

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

## 2.8 一个 PASS 的四种状态 —— **在收敛结果到来之前**先定死

**这一节是预先登记的,写在 m1 的 `--keep` 六个 validator 结果之前。** 理由是本包自己
的规矩:**一条需要人记得的规则会衰减,一条规定「产物必须含什么」的规则不会。** 等结果
落地再来讨论「这算不算真的通过」,是在盯着一个绿色结果的压力下做分类;先把判据写死,
分类就变成查表。`RUN-PLAN.md` 的「Predictions, before the run」是同一个做法。

**四种状态,由下往上,每一级都严格强于下一级:**

| # | 状态 | 判据(可查) |
|---|---|---|
| 1 | **在 mock 数据上通过** | 有 verdict `result: true`,输入来自封存语料 |
| 2 | **在一件被重放的产物上通过** | 同上,且该产物由 `mock_stages` / `replay_root` 提供;**m4 的两个 validator 从此永久封顶在这一级**(见 §3 m4) |
| 3 | **在一条真实的链上端到端通过** | 同上,且它评判的那个 kind 由**本次运行中真实执行的 stage** 产出 |
| 4 | **③ 且已知会拒绝坏输入** | ③ 成立,**并且**它在 `3885050` 的负控制里拒绝过注入的破坏 |

**只有第 4 级让一个 PASS 携带信息。** 前三级说的都是「这次没反对」;第 4 级说的是
「它会反对,而这次它没有」——**而这正是整件事想买的东西。**

**目前的分布(2026-09-05T13:xx,结果到来前):**

- **【2026-09-05T13:xx 登记值,结果到来前;当前值见 §2.9】第 4 级:0 个。**
  本包至今没有任何一个 validator 达到过。
- **【2026-09-05T13:xx 登记值,结果到来前;当前值见 §2.9】第 3 级:0 个**——`-noval`
  的运行里全部是 `check_nothing`;§2.2 的收敛运行是真实 validator,但喂的是语料中段
  (node=`mock`)。
- 五个已被负控制证明会判分的(§2.1)停在第 1–2 级,**它们只差第 3 级这一步**。
- 五个 m5 的(§2.6)连第 1 级的负控制都还没有。

> **上面两条为什么把限定词塞进加粗句里(2026-09-05 晚补,m3 诊断)。**
> 原来的写法是「加粗的数字」+「上一行的普通体日期」。**加粗、带数字、自足的那句
> 就是可引用单元,而限定词不在里面**——于是它经四个人转手,数字走了、上一行留在
> 原地,`第 4 级:0 个` 被当作当前状态报给用户两次。
> **这是 `tail -1` 的散文形态:原因就写在被保留的那一行的正上方。**
> **不是文档缺口,是排版缺口——加粗的东西才会被带走。**
>
> **这次改动改了什么、没改什么:** 只把限定词移进可引用单元的边界内。
> **零个数字变化,零个级别归档变化,快照仍然冻结。** 这和「为了保持最新而回填基线」
> 是两回事——后者今天上午已被拒绝,且理由不变。**记在这里,是因为一份预先登记记录
> 的可信度来自「改过什么是可查的」,而不是来自「从没被碰过」。**

**并且要记住这条 0 → 1 是什么:预先登记成功了,不是记录出错了。**(m3 的措辞。)
判据和分布在 13:xx 冻结,结果随后到达,§2.9 把它们归档到一条**事后谁也搬不动的杠**上。
**`check_command_parses` 到达第 4 级,是一个不可能被谈判出来的结果**——计数从 0 走到 1
正是这套机制在起作用。**今天关于这个数字的错误,全部发生在引用它的路上,没有一个发生
在产生它的地方。**

**所以 m1 的六个回来时,唯一要问的是:它评判的那个 kind,是不是这次运行里真实跑出来
的?** 是 → 第 3 级;而如果它同时在 §2.1 的五个之内 → **第 4 级,本包第一次。**

**不要把「六个全绿」写成一句总结。** 逐个填进 §3 对应阶段的条目里,因为那是读者先
撞上的地方——这是本文件自己的发现。

---

## 2.9 收敛结果,按 §2.8 的判据归档 —— **一个到了第 4 级**

**run `20260905T141058-6735d9`(m1 的 k6m1),`--package e2e-flow-keep-m1`,
`--var mock_stages=m2,m3,m4`。217 卡 4-7,与那次 78 分钟的基线同节点、同卡、同语料,
**只改了一个变量**。运行进行中(已过 13 个 task,在 `integrate_and_verify`);
下面是我从 `validation.yaml` 读出来的,不是从日志。**

**32 条 verdict,零拒绝。** 其中 12 条是 `check_nothing`,20 条是真实 validator。

> **更新(运行继续之后):不再是零拒绝。** 我重数:39 条 verdict,
> **`check_environment` 13 PASS / 1 FAIL**,唯一一条拒绝落在 **`integration_report`**
> 上——**那是真实 m5 产出的、这个 kind 第一件被它看过的真产物。见 §2.10。**
> 上面的级别归档不变(判据 3 问的是「评的 kind 是否真跑出来的」,那仍然成立),
> 但「零拒绝」这句话已经过期,保留原文并在此更正。

**关键事实:`mock_stages=m2,m3,m4` ⇒ stage 1 和 stage 5 真实执行,2–4 是重放。**
所以 §2.8 判据 3 要逐条查「它评的那个 kind 由哪个阶段产出」——**查 kind,不是查
validator 名**:

| validator | 评过的 kind | 真实执行? |
|---|---|---|
| `check_command_parses` | `patch_overlay` (m5) | **是** |
| | `profiling_evidence`、两个 `bench_result`、`profile_result` (m2) | 否 ×4 |
| `check_environment` | `deploy_kit` (m1)、`patch_overlay` (m5) | **是 ×2** |
| | m2/m3/m4 的九个 kind | 否 ×9 |
| `check_identity_resolved` | `operator_identity` (m3) | 否 |
| `check_kernel_table` | `profiling_mode_on.kernel_table` (m2) | 否 |
| `check_profiling_evidence` | `profiling_evidence` (m2) | 否 |
| `check_worklist_shape` | `kernel_worklist` (m3) | 否 |

### 归档结果

| validator | 达到 | 为什么不是更高一级 |
|---|---|---|
| **`check_command_parses`** | **第 4 级** | 判据 3 由 `patch_overlay` 满足,且它在 `3885050` 的负控制集内 |
| `check_environment` | **第 3 级** | 判据 3 满足(两个真实 kind),但**它不在负控制集里**,所以够不到第 4 级 |
| `check_kernel_table` | 第 2 级 | 在负控制集内,但只评过重放产物 |
| `check_identity_resolved` | 第 2 级 | 同上 |
| `check_profiling_evidence` | 第 2 级 | 同上 |
| `check_worklist_shape` | 第 2 级 | 同上 |

**`check_command_parses` 是本包第一个到达第 4 级的 validator:在一条真实的链上端到端
通过,并且已知会拒绝坏输入(注入语法错误、去掉可执行位,两次都拒绝,还原后通过)。
它的这次 PASS 携带信息,而不只是「这次没反对」。**

**「六个全绿」这句话是错的**——六个里只有一个到了第 4 级,一个到第 3 级,四个还在
第 2 级。**四个停在第 2 级不是它们的问题,是这次运行没有喂给它们真实产物**:
m2/m3/m4 被重放了。要把它们推到第 3 级,需要一次 m2/m3 真实执行的运行。

### 两条必须一起读的限定

**一、`check_environment` 的第 3 级是「按判据成立」,不是「有分量」。** §2.5 已经
确立它按构造不接触任何节点(m1 grep,零 subprocess)。**leader 20 分钟前的实例**:
m5 的 kit 记录 `node: crsuse2-m2m-237` 配 `node_ip: 10.245.144.69`——**那是 088 的
地址**。内部自洽、被每一份下游记录继承、而且是错的,`check_environment` 会全部放过,
因为各方一致。**这是 §6.1 的形状,又一个新鲜实例。** 所以它的第 3 级只是说「它评的
产物是真跑出来的」,不是说「它查出了什么」。

**二、判据是预先登记的,所以现在按原文用,不临时收紧。** 我可以主张
`check_command_parses` 评的只是一个 `items/command` 脚本能不能解析、这算不算「端到端
通过」——**但判据在结果落地之前就写死了(§2.8),而在看着第一个结果时提高门槛,正是
预先登记要防的那件事。** 如果这条判据太松,那是另一场讨论,不能用第一个结果来悄悄
解决。

*(运行仍在进行,m5 的其余 kind 还会带来更多 `check_environment` /
`check_command_parses` 的**真实**条目——那会增加计数,但不改变上面的级别归档,
除非出现拒绝。)*

## 2.10 `check_environment` 拒绝了它见过的第一件真产物 —— 而这提出一个我不该自己回答的问题

**m1 发现,我逐条复核。** 同一个运行 `20260905T141058-6735d9`:

```
check_environment   13 PASS / 1 FAIL
唯一拒绝            integration_report(真实 m5 产出)

mocked integration_report (20260905T091728)  items: env report.md schema text.json  -> PASS
real   integration_report (6735d9)           items:     report.md schema text.json  -> FAIL
```

**这是真阳性,不是 validator 的缺陷。** 真实的 m5 生产者**不写**
`items/env/environment.yaml`;封存语料里那份也没有;**是 mock adapter 造了一个出来。**
所以**本项目历史上这个 kind 的每一次 `check_environment` 通过,都是对着一件 mock 发明
出来的 item 通过的——而它一见到真产物就拒绝了。**

**两个会被记错的地方**(m1 的,我复核过):

1. **只有 `integration_report` 失败。** `done` 行还点了 `patched.measurement` 和
   `stock.measurement`,但我读了三份 `verdict.json`,两个 measurement 在两个 validator
   下都是 `true`。**它们是连坐**——任务的输出集是一起判的。**记三条就多记了两条。**
2. **严格包里这个 kind 上也挂着 `check_environment`**(`[check_environment,
   check_no_regression]`),所以这不是 `--keep` 包的产物,**全验证运行会一模一样地撞上。**

### 它提出的问题,以及我为什么不自己回答

§2.8 的第 4 级判据写的是:**③ 且在 `3885050` 的负控制里拒绝过注入的故障。**

`check_environment` 现在**拒绝过一件真实的坏产物**——按「已知会拒绝坏输入」这个**意图**
来说,这比一次注入更强。但**按判据的字面**,它不在 `3885050` 里,所以它仍然是第 3 级。

**我不在这里改判据。** 理由就是 §2.9 里写过的那条:判据是在结果到来之前定死的,而
**此刻改判据的方向恰好是「让我的文件多出一个第 4 级」**——这正是预先登记要防的那种
调整。**这是一个该由 leader 裁的问题,不是一个我顺手解决的问题:**

> **一次针对真实缺陷的拒绝,能不能替代一次负控制?**
> 支持:它证明了同一件事,而且是在真材料上。
> 反对:负控制是**受控**的——你知道注入了什么、也知道它该拒绝什么;一次真实拒绝
> 只证明它对**这一个**缺陷敏感,不告诉你它对别的缺陷是否敏感。

**在裁决之前,`check_environment` 在本文件里仍然是第 3 级。**

#### 裁决(leader,2026-09-05):**不能替代。`check_environment` 维持第 3 级。**

**理由一,就是上面「反对」那条:** 负控制是**受控**的——你知道注入了什么,所以你知道
一次拒绝证明了什么。**一次真实拒绝只确立它对「这一个」缺陷敏感,对其余的一言不发。**
第 4 级衡量的是**被刻画过的敏感度**,不是有用性。**论有用性,这次拒绝胜过我们所有的
负控制;论第 4 级要量的那件事,它是沉默的。**

**理由二,即使第一条可争,也足以这样裁:** 这个问题是**在看着第一个数据点时**提出的,
而改动的方向是「我的文件多出一个第 4 级」。**为此松口,等于教全队「结果够好时梯子会
移动」——那比任何一次单独的错误分级都更糟。**

**并且明确不做的事:不为「拒绝过一个真实缺陷」新增一个标记。** 那是同一个调整换一顶
帽子——**在一件仪器产出第一个读数时重塑它。** 这个事实已经写在上面的散文里,**今天它
就该待在散文里**。要不要单列一个字段,等到有若干个这类案例、且没有任何一个正在施压
的时候再决定。

*(记在这里而不是只留在消息里:一个只存在于消息中的决定,日后读起来像疏漏。)*

### 这一条同时给「不写 report」标了价

`check_environment` 是**最后一个一行 report 都不写的 body**(§「零理由」:0/469)。
这次拒绝的 escalation 只说了 *"output_validation did not pass"*,`message` 是通用的,
`detail` 一个字段名都没点。m1 拿到那句一行的原因,花了:三次 `verdict.json` 读取、
三个 handoff 的目录列举、两份 environment 记录的 diff(为了排除
`compare_fixed_across_inputs`)、以及一次跨运行对比 `p4_f`——**大约十五分钟。**
**一份 report 会立刻说「没有 `items/env/environment.yaml`」。**

**这就是「不写 report」的价格,而且它恰好在这个 body 第一次真的拒绝了什么的时候到期。**

## 2.11 一个绿不确立「换成 agent 跑也会绿」——四份 brief 里有一份一节都没点

**先说这一节不解决什么,免得下面那张表读起来像结论:**

> **移除触发器不等于关闭这一类。** 就算每一份 brief 都点全了每一节,下一次在
> `kind: ai` 阶段上发生的、与此无关的封存拒绝**仍然不可恢复、仍然是同样的沉默**——
> `output_absent` 在 agent 退出前不可见,而 `instruct()` 没有任何循环能够到一个已经
> 结束的 agent。**点全节名只是拆掉一个触发器,机制原样留着。**

**机制。** `agent_sys/handoff/content.py:62-87` 按 **content type** 规定 README 必须
有哪些小节,**不是一套通用的三节**(我最初被问的就是这个通用版本,它只对产出 `code`
的叶子成立):

```
reproducible     Purpose · How to run · Result · Environment · Watch out   (5)
code             Purpose · Interface · Boundary                            (3)
structured_text  Purpose · Schema                                          (2)
text             Purpose                                                   (1)
```

一个 `kind: ai` 叶子的 brief 若不点名这些小节,就要靠 agent 自己猜出**准确的标题**。
猜岔了就封存拒绝,**而拒绝发生在 agent 已经结束之后**。

**四个默认就是 `kind: ai` 的叶子**(`agent:` 行的默认值,不是被 `--var` 改出来的):

| 叶子 | 产出的 kind → 类型 | 需要 | brief 点了吗 |
|---|---|---|---|
| `deploy_and_prove` (`m1_deploy.yaml:325`) | `deploy_kit` → code | 3 | **点了** — `assets/deploy_and_prove.task/readme.md`,搜 *"It needs all three of"* |
| `build_workset` (`m3_analysis.yaml:443`) | `operator_workset` → code | 3 | **点了** — 搜 *"Purpose**, **Interface**, **Boundary"*,并带一条验收 |
| `optimize_kernel` (`m4_kernel_opt.yaml:473`) | `kernel_optimization` → code | 3 | **点了** — 三行对照表,且附了 `p5_fullreal_093c47` 那次真实拒绝的原话 |
| `integrate_and_verify` (`m5_integration.yaml:596`) | 两个 measurement → reproducible;`integration_report` → structured_text | 5 + 5 + 2 = **12** | **一节都没点** |

**`integrate_and_verify` 是唯一一个零,而它要求的最多。** 323 行的
`assets/integrate_and_verify.task/readme.md` 里,`How to run` / `## Result` /
`## Schema` / `## Purpose` 全部为零;唯一会被 grep 命中的是 `:140` 的
`*Watch out:*`——那是一句讲 bring-up 会刷 `Health check failed` 的散文,不是在指示
agent 写一个 `## Watch out` 小节。

**为什么它还没发作,而 m4 的发作了。** m5 那一阶段封在 `6735d9`,**一个 agent 即兴了
一次、恰好对了**;m4 的被**两个节点上的两个 agent**用不同的方式即兴,于是撞上。
**「至今没有」在这里只等于一次抽样。**

**另有一个潜伏项,不在上表**(它今天不是 `kind: ai`):`m2_profiling.yaml:321` 声明了
一个叫 `e2e_profiler` 的 `kind: ai` agent,而三个叶子(`:424`、`:446`、`:478`)全都
默认 `${m2_agent:-runner}`。**一旦有人传 `--var m2_agent=e2e_profiler`,三个叶子同时
变成 agent 体,而三份 brief 一节都没点。** 这类东西的发作时机,正是「某人传了一个
看起来无害的 var」的第一次。已直接交给 m2。

*(审计只读不改:每份 brief 都是它自己 owner 的文件。m5 的那份和 m2 的三份已分别路由。)*

## 2.12 m4 的 53 条离线 PASS —— **按判据它们不是第 3 级,而我不在这里改判据**

**材料(m4 判分,leader 转达,我没有独立复算计数):** 22 件产物、三次真实运行
(`p9`/093 m2、`217d`/217 m3、`093e`/093 m4),全部 `mock_stages=none` + `-noval`,
用 validator 自己的 body 离线判。**53 PASS / 57 可判**;4 条需要卡而弃权
(`check_workset_runs` ×2、`check_speedup_substantiated` ×2),3 条需要一次真部署
(`check_deploy_serves`,被 m4 按规矩 10c 拒绝执行,**不在那 57 里**)。

### 先更正一处:第 3 级从来不是零

**「第 3 级第一次非零」这句话过期了,而且是被我自己的 §2.9 过期掉的。** §2.8 里
「第 3 级:0 个」是**预先登记时的快照**(13:xx,结果到来前),它按设计不该被改写;
**§2.9 已经把 `check_environment` 归档为第 3 级、`check_command_parses` 归档为第 4 级。**
读者若只读 §2.8 的分布会读到过期数字——这是预先登记的固有代价,**修法是保留快照并
在此指路,不是回填它。**

### 判据怎么说,以及我为什么按字面用

第 3 级的原文是:**「② 且它评判的那个 kind 由*本次运行中*真实执行的 stage 产出」**,
这一级的名字是**「在一条真实的链上端到端通过」**。

**m4 这 53 条把「判定」和「产出」分到了两处:** 产物来自三次真实运行,而那三次是
`-noval`——**它们运行时没有任何真实 validator 在场**;verdict 是事后在 m4 的 zone 里
补判的。

- 按**目的**(把「真跑出来的产物」和「语料」分开):满足。产物确实是真实 stage 产的。
- 按**字面**(「本次运行」= 判定与产出同属一次运行):**不满足。**

**判据写下时没有设想过这两者会分离**,所以它在这个案例上是**有歧义的**,而不是明确
包含或明确排除。**而歧义的解法方向恰好是「让我的文件多出 53 个第 3 级」。**
今早的裁决(§2.10)拒绝了同样形状的一次放宽,**理由对放宽和收紧同样成立**:
**在看着第一个数据点时移动梯子,正是预先登记要防的那件事,方向不重要。**

> **所以:这 53 条在本文件里按第 2 级以上、第 3 级未达记录,理由是「判定不在产出的
> 那次运行内」。措辞问题上交 leader,和 §2.10 走同一条路。**

#### 裁决(leader,2026-09-05):**按字面。53 条不达第 3 级,缺口点名而不掩盖。**

**并且这一级的名字本身就够判,不必动判据正文:「在一条真实的链上端到端通过」——
离线补判按构造就不是端到端。** leader 明确没有说「目的解读是错的」,而是说
**「能解决歧义的那个解读,恰好是给我的文件加 53 行的那个,所以它必须等」。**

**重开的条件,写在这里而不是留在消息里:**

> **当这个措辞问题上不再挂着一份待归档的成绩时,它回来。**

**这不是拖延,而是这个决定唯一值钱的条件。** 一个没有触发条件的「暂缓」,过两天读
起来和「忘了」没有区别——**本文件对「永远」那条的处理是同一个道理:限定词必须可查。**

**并且明确不做:不为「离线补判真实产物」新增一级或新增标记。** 那是 §2.10 已经裁过的
同一个动作换顶帽子——**在一件仪器产出第一个读数时重塑它。** 这个事实待在散文里。

### 但有一条不依赖级别,而且比级别值钱

`check_trace_coverage` **重新解析了 trace**:3408762 events、205947 GPU kernels、
**4 个 rank 上 823736 条 GPU kernel event,与 manifest 一致**——**和 m2 独立到达同一个
数**。这正是 §2.7 的判据:**一个只有真的读过输入才能算出来的数,静默失败伪造不出来。**
它在两套谁都没共享的 harness 上各出现一次。

**一处必须自己泼的冷水:** 两次到达不等于两个**方法**——两边大概率都是解析同一份 trace
文件。按我今早自己赔出来的那条(*「说两个来源吻合之前,先问它们用的是不是同一个方法」*),
这**是同一个方法的两次独立执行**。它能排除「抄来的值」和「过期的值」,**排除不掉
「两边都错解了同一份 trace」**。§2.7 要的正是前者,所以判据成立;**只是不要把它说成
方法独立。**

### 分母

**53 / 57,永远不要写 53 / 60。** leader 的抵制是对的,而且还可以更准一格:
**57 本身也不是「全部」**——4 条弃权在 57 之内,3 条 `check_deploy_serves` 在 57 之外。
**七条缺席的理由本身就是发现**(两条「这里测不了,而占卡是别人的」,一条规矩 10c),
把它们折进分母就是把发现折没了。

### 第 4 级仍然是零,而这是 m4 自己说的

**这批里没有任何一条拒绝过注入的故障。** m4 未经提示就写了这句——
**在一份 53 比 4 的结果里主动指出它买不到第 4 级,是这批材料里最可信的一句话。**

*(另记:m4 在报任何 verdict 之前先找出了自己 harness 的两个缺陷,其中一个在 m2 已经
通过的产物上造出**四次自信的误拒**。他们的原话:*「我的 harness 里没有任何东西发现了它——
是和他们结果的矛盾发现的。」* **抓住它的是一个已知良好的对照,不是仪器**——这条属于
`CLAUDE.md` 的「不要相信工具」,不属于本文件的级别归档。)*

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

#### 一个我们**主动选择**的天花板,不是缺陷 —— 2026-09-05,用户指示

用户的理由是**「单次耗时太长了」**,而当天的测量支持它:m4 的一次 campaign 在
`forge_max_hours=1.0` 的预算下**光准备就花了 113 分钟**,三个 marker 缺失,**没有一个
源文件被改动**。于是 `CLAUDE.md` 规则 18:**stage 4 尽量重放,不重跑**
(`--var mock_stages=…,m4,…`,或 `--var forge_mock=1`)。

**这条决定给这两个 validator 划了一道永久的上限,应该在这里写清楚,而不是以后被当成
缺口发现:**

> **在本轮里,`check_optimization_shape` 和 `check_speedup_substantiated` 只可能
> 对着一件被重放的产物行使。** 它们「拒绝过 5 次 / 7 次」是真的,但那些拒绝此后
> 不会再增加新的**真实** campaign 的样本。

**而其中一条检查会因此**按构造**被满足**,这一点是 m4 读出来的,我复核过代码:
`_check_against_snapshot` 拿文档里的 `operator` 去它**随身携带的**
`workset.snapshot.yaml` 里查——**重放时这两半来自同一份语料,必然一致。** 所以在规则 18
之下,这条检查对我们永久落在「一个来源劈成两份」那一类里(`SKIP-AHEAD.md` §6.1 第四条)。

**仍然会咬人的是 `_cross_check`**(`check_optimization_shape.validator/check.py:536`):
它在**别处真的 stage 了一份 workset** 时,拿快照去和那一份比。在 m4 自己的 output
phase 里没有第二份,所以它推迟到 **m5 的 input phase**——那里两份都在。**而它在没能比
的时候会写下一条 note 说自己没比**,所以这个上限是**可见的**,不是隐形的。

#### premise gate 全绿,而且**一条 warning 都没有**,仍然没有保护那次比较

**这是那唯一一次真实 campaign 的产物**(run `20260905T104942-e6bd81`,我复核:
`forge.mock: false`、`ran: true`,所以它就是上面说的「今天已经回答过一次」的那一次)。

```
premise.verdict = {"held": true, "aborted_on": [], "warnings": []}
```

**不是「通过但有保留」,是通过且什么都没记。** 而同一次运行里,同期重测的 stock 比它
被记录下来时**慢了 7.1%**,因为相邻四张卡上有另一条线在跑。于是同一个 kernel、同一台
节点,出现两个分处「算不算改进」两侧的数:

```
对 workset 记录的 baseline   0.9883   (慢 1.2%)
对同期 stock                 1.0587   (快 5.9%)
```

**gate 全程是绿的,而且它不可能不是**——它看
`gpu_arch / gpu_count / tp_size / dtype`,**看不见机器忙闲,而忙闲决定了答案。**

**这和 `_agree_or_die` 只覆盖 28 个字段里的 3 个是同一族**(§2.5):
**一个守卫覆盖的是「有人想到过」的字段。** 区别只在于这次没被想到的那个变量不是字段,
是节点的负载。

#### 好的那一半:阶段自己拒绝了下结论

**两个读数都远低于 workset 声明的 `noise_floor: 1.1729`,于是 agent 整块删掉了
`claim`,两张测量表都留着。** 我复核过产物:

```
evidence.performance keys : ['entrypoint', 'protocol', 'baseline', 'measured']
claim present             : False
```

它自己的话:**「我删的是断言,不是数字。」**

**一个阶段在环境不利的那次运行上拒绝提出主张,正是本文件想要奖励的行为。**
§2.7 的判据说「生产者算出来、依赖它必须读过的输入」的内容才是证据;**这里补上对称的
一条:一个生产者在噪声高于信号时收回主张,和它算出一个数一样,是可核对的诚实。**

#### 归因:是分配,不是 agent

**争用来自「一台机器两条线」的分配决定,不是这次运行的过错。** 那条规则本身没有错,
但它有代价,**而代价恰好落在会产出「测量」的那些阶段上。** 要点名一个原因的话,
点名这个,不要点名 agent。

**推论,写给下一个人:凡是产出测量的阶段,它的前置条件里应当有一条「节点安静」,
而 premise gate 里没有这一条。**

**真实 stage 4 从此只用来回答一个问题:「一次真实的 stage 4 会做什么」。今天已经回答
过一次。**

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

---

## ESTABLISHED 2026-09-06 01:35 — `check_environment` on a real `integration_report`

**Carried as NOT ESTABLISHED across four rounds. Settled by `r5m1c`
(run `20260906T000840-bc92a6`, node 217, whole node, `mock_stages=m2,m3,m4`).**

**16 of 17 validators pass. The one refusal is structural, not a defect.**

```
handoff 089c6a06 (integration_report)
  items:  env  report.md  schema  text.json      <- env present, 1 environment.yaml
  record: node crsuse2-m2m-217 · container yihou_r5m1c_sgl_r5m1c-a
          started_at 2026-09-06T00:20:43Z        <- THIS run, this node

  check_environment    True    materials_files=149
  check_no_regression  False   materials_files=149
```

**Why this is worth something and the four earlier greens were not:** before tonight
every stage 5 in every rung was mocked, and `mock_m5.sh` rendered the environment
record itself. `compare.py` **has never written one** -- so this kind could not
satisfy a `strong` validator covering all fifteen kinds, and **the hole was hidden by
the mock rather than exposed by it.** m5's `8f05a94` fixed the producer; this run is
the first evidence the fix works on a real chain.

**Checked to the refuting standard, not read off the tally.** One zone in this run was
handed **zero** files (`validation.87f3c042…`, no report) -- the fourth instance of that
fault and the first that cost nothing. **It judged neither of these handoffs**; the zone
that did held 149 files. **A PASS establishes that the validator was invoked, not that
it saw anything -- so the file count is part of the claim, not a footnote.**

### The seventeenth: `check_no_regression`, refusing correctly

```
the stock arm does not reproduce m2's profiling_mode_off bench within 10%:
inter_token_latency (avg) -29.3%. The two stages measured different machines,
or one machine in two states.
```

**Content-derived, so it read the files. And it CANNOT pass in a corpus-middle
configuration** -- m2's bench comes from the corpus (another machine, another day)
while the stock arm ran tonight. **Round 4 measured -26.1%, round 5 -29.3%: two
independent measurements of the same structural gap.** Making it pass needs a real
m2, not a fix.

### Also promoted by this run

- **`check_measurement_order`: 2 PASS** on `stock.measurement` and `patched.measurement`
  -- it was the last entry standing at "unknown, not failing".
- `check_acceptance`, `check_bench_report`, `check_bench_result`, `check_command_parses`,
  `check_patch_live` pass on both arms.

### Not understood, flagged rather than diagnosed

**`stock.measurement` and `patched.measurement` are marked `invalid` although every one
of their validators passed.** Possibly sibling propagation from the failing
`integration_report`, possibly not. **Unread; nobody should quote a cause for it.**
