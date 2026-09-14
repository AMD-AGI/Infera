# 跳级调试（skip-ahead）——操作手册

把一次已完成运行里已封存的 handoff 物化成一个 `mock_root`，让调试运行跳过已经稳定
的前几个 stage，只把算力花在还没稳定的那一个上。

用户的原话就是这个机制的规格：

> 前面的 module 如果连续 3 次以上稳定运行了，再往后调试运行就跳过前面若干个
> module，从最后那次运行里拿 handoff 塞给还没稳定的那个阶段。

实现是 `assets/lib/replay_root.py`（m5）。本文件是它的操作手册；**规则、度量和拒绝
理由都在那个文件的 docstring 里，有分歧以代码为准。**

---

## 0. 先读这一句

> ## **这是调试加速器，绝不是验收路径。**
>
> 用户同样说得很明确：**整个调试完了的最后最后验收阶段，还是要求 e2e 全部真实跑通
> 一次。**
>
> 一次绿色的跳级运行，**只证明它真正跑过的那些 stage**，对被 replay 的那些什么都
> 不证明——那些是在它们产出的当天被证明的，仅此而已。工具写进 `PROMOTION.json` 的
> `ACCEPTANCE` 字段说的是同一件事，它每次运行结束也会在 stdout 上再说一遍。
>
> **如果你正在读这份文件是因为你动了"就拿这次跳级运行验收吧"的念头——这段话就是
> 为了拦住这个念头而写的。**

## 1. 今天还用不上，这是实测

2026-09-04 对 run root 下全部 41 次运行跑 `--list`，逐字输出：

```
unstable deploy_kit                       2 consecutive of 5 that reached a verdict;
                                          broken by 20260904T125637-e1ddf6
                                          | streak 2 (of 3) | 4 valid / 5 reached a verdict
                                          / 11 never sealed / 25 mocked
unstable operator_workset                 no run executed this stage for real
                                          (3 run(s) still generating, excluded), 21 mocked
unstable kernel_optimization              no run executed this stage for real, 13 mocked
unstable stock.measurement                no run executed this stage for real, 4 mocked
unstable patched.measurement              no run executed this stage for real, 4 mocked
unstable integration_report               no run executed this stage for real, 4 mocked
unstable kernel_worklist                  cannot tell — 24 run(s) with no recorded discriminator
unstable operator_identity                cannot tell — 24 run(s) with no recorded discriminator
unstable patch_overlay                    cannot tell — 4 run(s) with no recorded discriminator
unstable profiling_evidence               cannot tell — 24 run(s) with no recorded discriminator
unstable profiling_mode_off.bench_result  cannot tell — 25 run(s) with no recorded discriminator
unstable profiling_mode_on.bench_result   cannot tell — 24 run(s) with no recorded discriminator
unstable profiling_mode_on.kernel_table   cannot tell — 24 run(s) with no recorded discriminator
unstable profiling_mode_on.profile_result cannot tell — 24 run(s) with no recorded discriminator
```

（`e2e_packup` 一次都没被产出过，所以列表里没有它。14 + 1 = 15 个 kind。）

三个可以立刻读出来的结论：

- 剩下 8 个 kind 工具**答不出来**，见 §5。

> **状态更新，2026-09-05 01:30（lead）。** 上面这一节写于 2026-09-04 晚，当时
> 它说*「stage 1 目前不够格：连续 2 次，门槛 3 次」*、*「今天没有任何一个 kind
> 可以被跳过」*、*「stage 2–5 从未真实产出过任何东西」*。**三句都已经过时**，
> 保留原文并在此更正，而不是原地改写——因为它们是当时的真实读数：
>
> - **stage 1 已够格**：`streak 4 (of 3)`，6 次 valid。打断连击的
>   `20260904T125637-e1ddf6`（max-bs 8 被正确拒绝）现在落在连击之前。
> - **跳级已经真跑过一次**：rung 2g（`20260905T003217-c44c92`）重放了 rung 2f 的
>   kit，stage 1 的产出段从 59 分钟压到约 3 分钟且完全没碰 GPU。重放的确实是真
>   kit（`start_worker.sh` 带 `--cuda-graph-max-bs-decode`）。
> - **stage 2 的一半已经真实产出并通过**：rung 2f 的
>   `profiling_mode_off.bench_result` 三个 validator 全 true，其中
>   `check_bench_result` 拿到的是真实读数 `ceiling 32 >= decode concurrency
>   4.909`。
> - **另一半仍然没有**：`run_profiling_mode_on` 至今**没有产出过任何通过验证的
>   产物**（判据是 `validation.yaml` 是否存在，不是日志里的 `created` 字样）。
>   见 `temp/bugs/2026-09-05-a-program-task-is-marked-running-with-no-program.md`。
>
> **本文件其余部分未受影响**，尤其是顶部那条「绝不是验收路径」——它比写下它的
> 那一天更重要了，因为现在真的有东西可以跳了。

**门槛不要靠 `--allow-unstable` 绕过去做验收。** 那个开关是给"我知道它不稳，我就是
要拿它调另一个 stage"的场景用的，工具会把 `stable: false` 和原因一起写进
`PROMOTION.json`。

## 2. 两条命令

**第一条：只看，不决定，不写任何东西。**

```sh
python3 assets/lib/replay_root.py --list \
  --run /home/yihou/agent_sys_runroot/runs/<older> \
  --run /home/yihou/agent_sys_runroot/runs/<newer>
```

`--run` 可重复，**按时间从旧到新给**（连击是从最新往回数的）。加 `--kind K` 可以
只看某几个 kind，`--threshold N` 改门槛（默认 3，就是用户说的那个 3）。

**第二条：物化。**

```sh
python3 assets/lib/replay_root.py \
  --out /home/yihou/<something>/replay_root_<tag> \
  --node <你将要启动的那台节点> \
  --run <run> [--run <run> ...]
```

**`--node` 必填**，理由见 §6.1——那不是一个可选的谨慎，它是这个机制唯一真正生效的
同节点守卫。

**`--out` 不要指向 `/shared_nfs`。** `/shared_nfs` 从登录节点看是**只读**的，所以
`replay_root` 没法在登录节点上把 root 写进计算节点会去读的那个位置。两条路：在节点上
跑这个工具，或者物化到 `/home/yihou/...` 再拷过去。（m5 是在照着 leader 那句"写进
`ws_handoff_refine/`"做的时候撞上的。）

产物是 `<out>/<stageN-xxx>/<kind>/content/`，外加一份 `PROMOTION.json`——里面记着
生成时间、**完整的命令行本身**、门槛、被调查的运行、`required_node` 与 `kit_nodes`、
每个被提升 kind 的来源运行 / handoff id / 版本 / 文件数 / verdict / seam 答案，以及
**没被提升的 kind 和原因**。三周后读到一个 replay handoff 的人，手上只有这个文件。

**`mock_root` 是一个目录、供全部五个 stage 共用。** 一个只装了被 replay 那个 stage
的 root 会把其余每一个 mock 的 stage 全部弄坏——`mock: no such stage
.../stage2-profiling`。这对这个机制存在的意义是致命的：**前面跳过、后面 mock**，而
后面那些没地方读。工具现在会自动把其余 stage 播成指向封存语料的**符号链接**（所以
出处一个 `ls -l` 就看得见，而不是一份会陈旧的拷贝），`--no-seed` 可以关掉。
**手工拼装一个 root 的人必须知道这件事。**

**第三步：把它交给一次调试运行。**

```sh
--var mock_root=<out> \
--var mock_stages=m1,m2 \
--var m1_agent=runner --var m2_agent=... （每个仍被 mock 的 stage 一个）
```

注意 `mock_stages` 和 `m<N>_agent=runner` 是**两件必须一起做的事**：前者告诉
`mock.sh` 哪些 stage 走 mock 路径，后者把 `kind: ai` 的 leaf 换成会跑 `entry.sh`
的程序 agent。**默认值是真实 agent**，所以晋级一个 stage 靠的是**去掉**一个
`--var`，漏掉一个就是"模型被意外调用"——这已经发生过一次。

## 3. "稳定"现在的准确定义

**连续，不是累计。**（用户的词是"连续"。）工具从最新的运行往回数，规则是：

| 条件 | 说明 |
|---|---|
| 到达终态 | 运行必须真的走到一个 verdict。**被作业到期或被人杀掉的运行会留下 `generating` 且一行 `validation.yaml` 都没有**——进程死了，attempt 从没"结束"过，`_close_model_slot` 没跑。所以"被外部终止"不是推断出来的，它就是 `generating`，工具直接排除，不计入也不算断连 |
| 同一组 validator | N 次运行必须被**同一组** validator 评判过，且每个 `result: true`。validator 组不同**不算低计数的稳定**，那意味着这些产物根本不是被同一把尺子量的。报告时带上分布，从不平均掉 |
| 被拒绝也算终态 | 一次封存并被拒绝的运行**是**一个 verdict，它计数、并且断连击 |
| `invalid` 的二义性有守卫 | `agent/runner.py:980` 在 attempt 结束时 slot 还开着的情况下也封 `INVALID`，和"validator 拒绝"无法区分（见 `temp/bugs/2026-09-04-invalid-means-two-things…`）。所以一个 `invalid` 只有在它的 validator 组**等于**通过运行上看到的完整组时才算终态；组短了说明 phase 没跑完，那次运行被跳过，**不算在这个 stage 头上** |

**"真实执行过"是独立于以上的第一道筛。** 这条是补一个真实存在过的缺陷：曾经有一份
调查报告 `deploy_kit` **STABLE，27 次运行**，而 stage 1 真正部署到 GPU 上的次数是个
位数。原因是**一个 mock leaf 拷贝的是一份此前已封存、已通过验证的产物**，所以它按
构造就会通过**同一组** validator——它就是当初通过那些 validator 的那份产物。27 次是
**一次确认被数了 27 遍**，而用一个从未执行过的 stage 去认证这个 stage，恰恰授予了
这个机制绝不能授予的那种跳过。

这是 CONTRACT §4.6 上升了一层：**validator 组分不出"真产物"和"真产物的拷贝"，因为
两边共用同一件产物。**

判据是：一个 stage 靠**去掉** `--var m<N>_agent=runner` 来晋级，而 store 会按 task
记下解析后的值（`store/task/*.json` 的 `agent_spec`）。实测：

```
deploy_and_prove  agent_spec=runner        26 runs   mocked
deploy_and_prove  agent_spec=e2e_deployer  15 runs   real
  其中 kit 同时封为 valid 的:                4 runs
```

比较用的是 **`!= "runner"`，而不是 `== 包里的真实 agent 名`**，这是决定不是偷懒：
任何人传 `--var m1_agent=<别的真 agent>` 也是真的执行了，拿包里的默认值做等值比较
会把它判成 mock。只有一个值意味着"这个 stage 没跑"，就是 `runner`。

## 4. 为什么 replay 一个 `deploy_kit` 是安全的

`SKIPPABLE` 表问的是每个 seam 一个问题：**下游读的是这件产物，还是它所描述的那个
进程？** 十五个 kind 里**目前只有 `deploy_kit` 有答案**，其余全是 `None`——工具会
照样物化并**明说**答案缺失，而不是猜，因为一条错的表项换来的是一次看起来像别人
stage 有 bug 的调试。

`deploy_kit` 的答案是 `True`，**而且是按构造成立的，不是靠意图**：

- `deploy_and_prove` 在 STEP 5 写出 kit，在 STEP 7 拆掉部署，**两步都在 m1 自己的
  task 里**。所以 `runtime.container` 在**每一次**运行里都是**立刻**就指向一个已死
  容器，不是 replay 之后才这样。kit 自己的 `runtime.notes` 就是这么写的。
  **replay 没有引入任何这张图不是每次都在经历的陈旧性。**
- `runtime.endpoint` **在任何地方都没有读者**——m1 之外的每一处出现都是写。
- **从消费者一侧确认过，不只是从生产者一侧推断。** m2 的 `load/line.sh:131-149` 只
  读 `fixed.{node,image,image_id,model_name,served_model_name,tp_size,gpu_devices}`
  和 `runtime.replayed_from`（一个不会过期的静态出处字符串），**别的什么都不读**。
- 让这件事可核对而不是断言的那点精度：`runtime.endpoint`/`container`/`ports`
  **确实**出现在 `line.sh` 里，但它们来自 `deployment.json`——**m2 自己的
  `deploy.sh` 刚刚写下的那份握手**（`:288`、`:300`、`:306`），不是来自被注入的 kit。
  `:263` 从 kit 里跑 `deploy.sh` 起自己的引擎，`:288` 在没写出握手时拒绝。

**kit 提供的是"怎么部署"，部署本身提供"往哪里发流量"。** 这是设计而非侥幸：M2.5
（`line.sh:19` 引了原文）——*"module 1 的 output 已经包含了如何部署的全量信息"*——正是
把 `serve_*` 和 `check_service_live` 从 stage 2 删掉的那条规则。

## 5. 局限，而且现在很大

### 5.0 最大的一条：跳过之后，后面的阶段只能继续 mock，不能 real

**测于 2026-09-05 03:20（lead），rung 2g 与 2h。** 这一条推翻了本文件其余部分
隐含的前提，所以放在最前面。

跳过机制是**通过把 stage 1 变成 mock** 来实现「跳过」的。而
`assets/deploy_and_prove.task/mock_adapt.sh:195-217` 在 mock 时会**故意**把 kit
的三个入口脚本换成 stub：

```sh
STUB="$PKG/assets/check_deploy_serves.validator/stub_kit"
for f in deploy.sh wait_ready.sh teardown.sh; do
  mv "$PACKUP/scripts/$f" "$PACKUP/scripts/sealed/$f"   # 真的那三个被保留
done
cp "$STUB"/deploy.sh … "$PACKUP/scripts/"               # 装上 stub
## MOCKED DEPLOYMENT ENTRYPOINTS — read this before trusting `scripts/`
```

**所以重放出来的 `deploy_kit`，它的 `scripts/deploy.sh` 是一个 stub。** 任何真的
要从它部署的下游阶段，拿到的都是 stub：

```
rung 2h  mode_on  ready after 1s      （真实冷启动实测 222 s / 232 s）
                  deployment up … in stub_yihou_e2e_flow_pmon
                  probe 404 → exit 1 → output_absent → escalated → 永久挂起
rung 2g  mode_off engine_argv.txt 内容 = "Error response from daemon:
                  No such container: stub_…"，因为 stub 只往握手里写了一个
                  容器名字符串，根本没有容器
                  ——而三个 validator 仍然全部通过
```

**「前面跳过、后面真跑」今天做不到。** 跳过之后，后面的阶段可以继续 **mock**，
但不能 **real**——这与这个机制被要求做的事正好相反，而且从外面完全看不出来，
除非去读暂存 kit 里的 `scripts/`。

**这不是缺陷，是设计。** `mock_adapt.sh` 做的正是它声明要做的事，还留了表头警告。
问题在于跳级机制建立在 mock 之上，于是继承了这个语义。

**补救办法存在而且很小**：真入口还在 `scripts/sealed/`。要喂给真实下游阶段的
重放必须还原它们——要么 `replay_root.py` 物化时反适配，要么消费方在
`scripts/sealed/deploy.sh` 存在时优先用它。**尚未实施**：一个是 m5 的工具、一个
是 m1 的适配器，选哪边是设计决定。

**§4「为什么 replay 一个 `deploy_kit` 是安全的」仍然成立，但要这样读**：作为
**配方**它是安全的（`fixed.*` 那些静态字段是真的，`runtime.container` 本来每次
就是死的）；作为**可执行的部署入口**它不是。两者的区别以前不重要，因为没人真的
从重放 kit 部署过。

### 5.1 工具答不出来的那些 kind

**十五个 kind 里只有六个能分辨真假**（来自四个声明了 agent 的 closure：
`deploy_and_prove`、`build_workset`、`optimize_kernel`、`integrate_and_verify`）。
**另外八个印的是 `cannot tell — N run(s) with no recorded discriminator`**
（第十五个 `e2e_packup` 一次都没产出过）。

原因：m2 的三个 closure、`identify`、`apply_patch`、`packup` 在包里就写死
`agent: runner`——对它们来说 `runner` **就是**真实路径，mock 是在 `entry.sh` 里由
`mock.sh` 读 `E2E_MOCK_STAGES` 选的，而**这个变量 run tree 不记录**（只有暂存 yaml
里那个未解析的 `${mock_stages:-all}` 模板活了下来）。`runtime.replayed_from` 也补不
上这个洞：`mock_adapt.sh`（m1）和 `mock_m5.sh` 会设它，**`assets/lib/mock.sh` 不设**
——所以它覆盖的正是 `agent_spec` 已经覆盖的那两个 stage，其余一个都没多。

**工具会低报，而且这是故意选的方向。** rung 2 里 m2 的 stage 确实是真的，而这张表
不会这么说。**请把这张表当作不完整的来读**，而不是当作一份完整清单。宁可让一个真的
stage 被判为"说不准"从而多跑一次，也不要让一个 mock 的 stage 被认证为稳定。

## 6. 工具对 environment 记录做的改写，以及一处它拒绝改的

改写只发生在 `--out` 下的**拷贝**上，**永远不动源运行**——一棵 run tree 是证据。
`--no-rewrite` 可以关掉它，但那是给"我想看这次运行到底产出了什么"用的，不是给调试
运行用的（关掉就会留下会让 m3/m4 拒绝的旧 jobid，和一个可能解析到别的进程的容器名）。

| 字段 | 处理 | 为什么 |
|---|---|---|
| `runtime.slurm_jobid` | **删掉（置空），不是改写** | `_agree_or_die`（`run_in_container.sh:96-104`、m3 `measure_in_container.sh:127-129`）在"环境里的值和记录里的值都非空且不同"时**退出 1**。replay 的 kit 带着旧 jobid，调试运行有新的，于是 m3/m4 报 *"slurm_jobid is 'X' in the environment and 'Y' in the record"*——读起来像启动参数配错了，而不像注入的副作用。**这是唯一一个会看起来像别人的缺陷的失败。** 而 `_agree_or_die` 在记录为空时会采用环境里的值，也就是真的那个；schema 的 `runtime.required` 是 `[container, endpoint, started_at]`，删掉这两个仍然通过校验 |
| `runtime.transport` | 同上 | 同上 |
| `runtime.container` | 改成 **一个解析不出来的名字** `replayed-from-<run>-NOT-RUNNING` | 容器名带 run tag，但**不统一**（2026-09-04 有一个 kit 用的是只有日期的 tag），所以一个 replay 的名字**可能解析到一个活着的、但完全是另一个进程的容器**。那时 `docker inspect` 成功、node/jobid/transport 全都一致，m4 会 exec 进错误的容器而每一个字段都校验通过。解析不出来的名字会走 m4 的 ephemeral 路径，从 `fixed.image` 新建容器、记 `mode=ephemeral`，并声明"在那里测出的加速比"和"在活部署里测出的"是两个不同的主张。**响亮地错好过安静地错**，而这里响亮的那条路恰好也是对的 |
| `runtime.endpoint` | 改成一个死标记 | schema 要求它存在，但没有任何读者 |
| `runtime.replayed_from` | 写上来源运行；**如果来源本身就是 replay，就串起来**（`A <- B`） | 一次 replay 的 replay 不能抹掉第一次。实测：rung 1 的 `deploy_kit` **本身**就带着 `replayed_from: …/cheat_for_mock/stage1-deploy/deploy_kit`，因为那次运行 mock 了 stage 1。覆盖掉会把"封存语料 → 运行 X → 这里"塌成"运行 X → 这里"，告诉读者这些数字是一跳之前一次真实 bring-up 来的，而它们根本从来不是 |
| `fixed.node` | **明确拒绝改写** | m1 建议过在节点变了的时候改写它。**不改，并且把理由说出来而不是默默照做**：那件被 replay 的产物真的是在旧节点上产出的，改这个字段会让记录声称一次测量发生在它没发生的地方。同节点这条约束改由 §6.1 的 `--node` 在**构建 root 时**强制 |

### 6.1 同节点是硬约束，而它的守卫**不是** `compare_fixed_across_inputs`

**这一条推翻了本文件此前的写法，也推翻了 `replay_root.py` docstring 曾经的写法。
m5 于 2026-09-04 晚实测（`270710f`、`5458dfd`）：**

```
replayed kit 里写的 node:   crsuse2-m2m-217
下游继承到的 node:          crsuse2-m2m-217
四个被比对的字段:            node, gpu_arch, image_id, model_path —— 全部一致
```

每一个下游 handoff 都用 `env_render --inherit <被 replay 的 kit>` 渲染自己的
environment 记录。所以那四个被比对的字段**全部是从 replay 里拷贝来的**，它们彼此
一致，比对通过——**在一台这次运行根本没有使用的节点上**。这就是 CONTRACT §4.6 那句
话的又一个实例：**一个比较检测不出所有参与方共有的故障**，而这里所有参与方继承的
就是同一份记录。

真正强制同节点的是 **`_agree_or_die`**（`run_in_container.sh:105`、
`measure_in_container.sh:127`），它拿环境里的 `E2E_NODE` 和 `fixed.node` 比——
**但只在一个真实执行的 stage 里生效**。而跳级恰恰是"前面跳过、后面 mock"，所以
**在这个机制存在的意义所在的那个配置里，根本没有任何东西检查节点。**

> **不要把 `compare_fixed_across_inputs` 写成这里的守卫。跨节点 replay 它是过的。**

#### 通往「自洽但不真」的第三条路:自指

**继承、铸造、自指——三条不同的路,同一个结果。** 前两条在上面和
`WHAT-GREEN-ESTABLISHES.md` §2.5 里:一份记录可以从被 replay 的 kit **继承**过来,
也可以在产出时就被**铸造**错。第三条是 leader 在 2026-09-05 手工构造 stage-4 产物时
遇到的,并且主动记了下来:

**`premise.run_environment` 被设成 workset 自己的 environment。** 因为没有单独的
campaign 跑过,所以没有第二个环境可以填。于是 `_agree_or_die` 拿这份记录去和……它自己
比,**按构造必然满足**。

**这不是一个应该修掉的缺陷,写在这里也不是为了要求谁去修。** 相反的做法是编造一个
不存在的 run environment,那正是本包每天在拒绝的东西(CONTRACT §5.3)。**产物说了
它做过什么,这一节说为什么那很弱**——两句都要在,读者才拿得到完整的一句。

**它弱在哪里,精确地说:** `_agree_or_die` 的价值来自「两个独立来源是否一致」。当两边
是同一个来源时,它仍然返回通过,但那次通过**不携带任何信息**——它没有排除任何一种世界
状态。和 §4.6 是同一条:**一个比较看不见所有参与方共有的故障**,而自指是「所有参与方」
退化成一个的极端情形。

**给读者的判据:** 看到一次 `_agree_or_die` 通过,先问**它比的两个值是从哪来的**。
两个独立来源 → 这次通过有信息;一个来源分成两份(继承、铸造、自指) → 这次通过只说明
拷贝成功了。

#### 第四条路:一份文档和它自己随身携带的快照,是一个来源不是两个

**m4 发现,2026-09-05,我复核过代码。** `check_optimization_shape` 的
`_check_against_snapshot` 拿文档里的 `operator` 去它**随身携带的**
`results/workset.snapshot.yaml` 里查,查不到就拒绝。听上去是一次交叉核对。

**但在 mock 路径上,文档和它的快照是从同一份封存语料里一起拷出来的。** 于是两者
必然一致——**内部自洽,而只在对上游时才是错的,而两边都不引用那个上游。**

**这就是同一个机制的第四条路。** 判据不变:*它比的两个值是从哪来的*——这里的答案是
「一份语料,拷了两次」。

**同一个 validator 里有一个做对了的对照,可以并排看:** `_cross_check`
(`check_optimization_shape.validator/check.py:536`)在**别的 handoff 里真的有一份
workset 被 stage 出来时**,拿快照去和**那一份**比,不一致就点名拒绝。它是
**opportunistic** 的,在 m4 的 output phase 里没有第二份可比,所以推迟到 m5 的 input
phase——那里两份都在。

**而它在没能比的时候会写下一条 note 说自己没比**(逐字):

> *"no workset staged in this phase, so the snapshot is taken on trust here;
> it is cross-checked in m5's input validation, which stages both"*

**这正是「缺席必须可见」那条,而且它在今天之前就已经被实现了。** 一个读 m4 输出验证
的人,现在就能看出这次交叉核对没有跑。

**所以关于「算子不一致」的正确说法比「没有 validator 在看」窄得多,也有用得多:**
一件写错算子的产物会被**产出、封存、并且走过一个阶段**,然后才被拒绝。**不是不可见,
是被推迟了**,而推迟是有实测理由的(绑到 m4 自己的 phase 上*"would fail an innocent
producer, which is measured"*)。

#### 这一条同时印证了 `WHAT-GREEN-ESTABLISHES.md` §2.7 的判据

同一次运行里两个机制都在场,而只有一个是证据:

- `_check_against_snapshot` 比的是**两份拷贝**——一次拷贝就能满足它;
- `stock_vs_m2` 算的是**一个数**,而那个数只有真的读了 m2 的 bench、又真的测了 stock
  臂才能得到——**一次拷贝伪造不了它。**

**「生产者算出来的、其值依赖于它必须读进来的输入」**,和「两个字段相等」,在
`-noval` 之下是完全不同量级的证据。

**现在的守卫是工具自己。** `--node <你将要启动的那台节点>` **必填**，root 里 replay
的 kit 来自别的节点就以 rc=2 拒绝；`--allow-cross-node` 是一个需要显式做出的决定。
`PROMOTION.json` 里记 `required_node` 和 `kit_nodes` 两个必填字段，这样一小时后的
读者不用打开任何 handoff 就能回答"我能不能在这里启动它"。

用户会看到的拒绝长这样：

```
replay_root: this root replays a kit produced on ['crsuse2-m2m-217'] and you asked
  to launch it on 'crsuse2-m2m-275'.
  Refusing: nothing downstream will catch this.
```

**这道拒绝在"构建 root"的时候生效，不是在"运行"的时候。** 用 `--node A` 物化完再
去 B 上启动，目前**没有任何东西会拦你**。补上它需要在 `mock.sh` 里加守卫，那是
leader 的共享文件，且在 rung 2e 期间被占用。所以请把它读成**"构建 root 时会被
拒绝"，而不是"不可能发生"**。

选 refuse 而不是 warn 的理由是 leader 定的，也是决定性的那条：一条在物化时打印的
warning 只会被**已经知道这件事的那个人**读到一次；而故障会在一小时后落到另一个人
头上——那个人正看着一个偏了 4 倍的 bench 数字去追一个内核回归。那就是
`--cuda-graph-max-bs` 的形状，它已经代价过一整天。

`replayed_from` 还有一道写回校验：`environment.schema.json` 的 `runtime` 是开着的
（`additionalProperties: true`），所以这个字段是未声明的，**拼错了会静默通过校验**
——实测用标准 `Draft202012Validator` 对真实的 rung-1 记录跑过：`replayed_from`、
`replayed_form`、`zzz_not_a_field` 三者**都是零个 schema 错误**，后两个读回来是
`None`。而 `None` 对全部五个消费者都意味着**"这不是 replay"**。所以一个字母打错，
就会让一份 replay 的 kit 被整条流程当成真实 bring-up——不安全的那个方向，来自一个
拼写错误，出现在那个唯一职责就是"声明这件产物不是它看起来那样"的字段上。工具因此
**用消费者用的那个取值方式把它读回来，读不到就拒绝**。

## 7. 工具的其它拒绝行为

- **绝不删除不是自己写的树。** 如果 `--out` 非空且里面没有 `PROMOTION.json`，直接
  拒绝并要求换一个 `--out`——`mock_root` 有可能就是那份封存语料，那不是我们的东西。
- 提升了一个 seam 答案未知（`SKIPPABLE` 为 `None`）的 kind 时，会在 stderr 上明确
  提示，并说：**在这个 seam 被回答之前，把消费 stage 里的失败当成"可能是这件事"而
  不是"那个 stage 的缺陷"。**

## 8. 动手前的清单

1. `--list`，逐行读完，**不要只看有没有 `STABLE`**——读 `streak N (of 3)` 和
   `broken by` 后面那个运行。
2. 打算跳过的 stage，检查它不是 `cannot tell`。是的话，这个机制没有为它提供证据，
   决定要自己做并写下来。
3. `--out` 指向 `/home/yihou/` 下一个新目录——**不要指向 `/shared_nfs`**，它从登录
   节点看是只读的（§2）。
4. **同一个节点，并且用 `--node` 把它说出来。** 守卫是工具的 `--node`，**不是**
   `check_environment` 的 `compare_fixed_across_inputs`——后者在跨节点 replay 上是
   过的（§6.1）。而且这道拒绝只在构建 root 时生效：物化完换台机器启动，没有东西
   会拦你。
5. `mock_stages` 和每个仍被 mock 的 stage 的 `m<N>_agent=runner` 一起给，一个都
   不能漏。
6. 跑完之后，把 `PROMOTION.json` 和运行结果放在一起归档。
7. **不要用这次运行验收。** 最后的验收是 `--var mock_stages=none` 的一次完整真实
   e2e。
