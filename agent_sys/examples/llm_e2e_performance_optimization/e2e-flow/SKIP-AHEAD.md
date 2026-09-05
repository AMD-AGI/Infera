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
  --out /shared_nfs/yihou/agent_sys/ws_handoff_refine/replay_root_<tag> \
  --run <run> [--run <run> ...]
```

产物是 `<out>/<stageN-xxx>/<kind>/content/`，外加一份 `PROMOTION.json`——里面记着
生成时间、**完整的命令行本身**、门槛、被调查的运行、每个被提升 kind 的来源运行 /
handoff id / 版本 / 文件数 / verdict / seam 答案，以及**没被提升的 kind 和原因**。
三周后读到一个 replay handoff 的人，手上只有这个文件。

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
| `fixed.node` | **明确拒绝改写** | m1 建议过在节点变了的时候改写它。**不改，并且把理由说出来而不是默默照做**：那件被 replay 的产物真的是在旧节点上产出的，改这个字段会让记录声称一次测量发生在它没发生的地方。让 `_agree_or_die` 和 `check_environment` 的 `compare_fixed_across_inputs` 去拒绝才是正确结果——它的含义是**跳级调试要求同一个节点**，这是这个机制真实存在的约束，写下来比糊过去好 |

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
3. `--out` 指向 `/shared_nfs/yihou/agent_sys/ws_handoff_refine/` 下一个新目录。
4. 同一个节点。`fixed.node` 不会被改写，换节点就会（正确地）被拒绝。
5. `mock_stages` 和每个仍被 mock 的 stage 的 `m<N>_agent=runner` 一起给，一个都
   不能漏。
6. 跑完之后，把 `PROMOTION.json` 和运行结果放在一起归档。
7. **不要用这次运行验收。** 最后的验收是 `--var mock_stages=none` 的一次完整真实
   e2e。
