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
