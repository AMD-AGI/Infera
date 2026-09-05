# `e2e-flow` — the five stages, in one graph

Deploy → profile → analyse → optimise one kernel → integrate, as **one
`agent-sys run`**.

The five stages also exist as five separate packages next door
(`../{deploy,profiling,analyze,kernel-opt,integration}-demo/`), each driven to a
real cluster run on 2026-09-02. They are not a flow: **a handoff only travels
inside one run's graph**, so five packages are five runs and nothing chains.
This package is the join the repo-root `mission.md` asks for. The five demos are
kept, untouched, as reference and as a fallback.

## Status — 2026-09-04, late

The contract is frozen, the graph loads, the bodies are written, and **the
ladder has reached rung 1** (`RUN-PLAN.md:1644`): stage 1 real, stages 2–5 still
replaying sealed handoffs. Rungs 2–5 have not run. See 中文报告 §5.

```
17 closures (11 leaves + 6 non-leaves) · 15 handoff kinds · 21 validators
```

## Read these, in this order

| file | what it settles |
|---|---|
| [`CONTRACT.md`](CONTRACT.md) | **the frozen cross-module contract** — the fifteen kinds, the naming rule, the environment rule, the schema rule, and what each module deletes |
| [`RUN-PLAN.md`](RUN-PLAN.md) | **the promotion ladder** — rung by rung, the vars that change with each, and a standalone verification section per module |
| [`MOCK-MAP.md`](MOCK-MAP.md) | which sealed handoff stands in for which kind, and the six adaptations that are real work rather than a copy |
| [`assets/schemas/README.md`](assets/schemas/README.md) | who writes which schema, and against which real artefact |
| [`../todo.md`](../todo.md) | everything the mission deferred, with what would settle it |

## Run it

Six variables carry no default, because they are facts about one allocation on
one cluster and a default would be one machine's answer shipped as everyone's.

```sh
python3 -m agent_sys.cli.main show \
  --package agent_sys/examples/llm_e2e_performance_optimization/e2e-flow \
  --var jobid=1 --var node=n --var node_ip=0.0.0.0 \
  --var model_name=m --var model_path=/p --var image=i
```

`show` loads and type-checks every yaml, derives the edge set from the handoff
wiring, checks it against every `froms`, and dispatches nothing — **in under a
second.** It is the loop; run it after every edit. `run --dry-run` is the next
rung, `run` with `--var mock_stages=all` the one after.

Promote one stage at a time out of mock — `--var mock_stages=m2,m3,m4,m5`, then
`m3,m4,m5`, and so on — so that a failure is attributable to the stage that was
just promoted.

## The shape

```
main
├── m1_deploy          deploy_and_prove                    → deploy_kit
├── m2_profiling       run_profiling_mode_off  ┐
│                      run_profiling_mode_on   ├─ parallel → profiling_evidence
│                      merge_profiling_evidence┘
├── m3_analysis        rank → identify → build_workset     → operator_workset
├── m4_kernel_opt      optimize_kernel                     → kernel_optimization
└── m5_integration     apply_patch → integrate_and_verify → packup
                                                           → e2e_packup  (is_end)
```

`deploy_kit` reaches all four later stages: it carries the environment record,
and m3 and m4 have to *run* things — a workset's tests, then KernelForge — in
the container m1 brought up.

## Three things a reader will otherwise assume wrong

1. **There is no `serve_*` task anywhere.** Bring-up and use may not be split
   across agents (M2.5, M5.2), so a task that needs a service brings it up in
   its own STEPS and tears it down. That deleted five tasks and eight handoff
   kinds relative to the demos.
2. **`items_schema` is not the schema layer.** For a file item it validates the
   *filename string*; the contents are never read
   (`handoff/content.py:184-197`). The schema layer mission G2 asks for is
   `assets/schemas/`, loaded from both sides by `assets/lib/schema.py`.
3. **The 5% / 10% regression bars are measured and must not be widened.** The
   within-arm round-to-round spread on a steady node is ~2%. A previous round
   widened them to 35% / 30% in response to a *cross-instance* artefact; the
   missing control is a comparability gate at bring-up, not a looser bar
   (`../todo.md` T7).

---

# 中文设计报告

本节是给评审用的设计说明，事实全部来自本目录下的 `main.yaml`、`shared.yaml`、
`steps/*.yaml`、`CONTRACT.md`、`RUN-PLAN.md`、`../todo.md`、`assets/*.validator/readme.md`
与 `assets/lib/` 的 docstring，并与 `python3 -m agent_sys.cli.main show` 的输出交叉
核对过（17 closures，与下表一致）。

## 0. 契约文件（CONTRACT.md）

跨模块的约定不在本 README 里，而在 **`CONTRACT.md`（1515 行）**——它是冻结的，
本 README 只是它的导读。它绑定的东西：15 个 kind 的清单与命名规则（§1）、
"每个 handoff 都带 environment"（§2）、schema 由 producer 与 validator 共读（§3）、
validator 的规则（§4）、运行时与集群纪律（§5）、"spec 里不得有站点事实"（§6）、
以及五个 owner 共用一个 worktree 时的提交规则（§8a）。

其中两节是这一天新加的，值得单独看：

**§4.6 —— 相对检查看不见两边共有的故障。** m5 用构造输入实测：
`stock_vs_m2` 比对在"两臂都健康"和"两臂都退化成 eager decode"两种情况下
**输出完全相同**（同样的字段、同样三个 `within_tolerance: True`），而后者的数字
错了 4.6 倍。改成"顺便比一下引擎配置"也无效——在出问题的那种情况里两边是一致的
（8 == 8）。**只有绝对门槛能抓到**：graph ceiling 必须不低于负载实际达到的并发。
实测同一节点、同一镜像、同一 tp=4、同一 1024/1024/conc-16 负载，只改一个 flag：

```
--cuda-graph-max-bs  8   ITL 42.15 ms    64 requests    312 tps
--cuda-graph-max-bs 16   ITL  9.31 ms   304 requests   1649 tps
```

decode 差 4.5 倍，而 `node`/`gpu_arch`/`image_id`/`model_path` 四个字段完全相同。
实现是 `assets/lib/graph_ceiling.py`，**一份实现、两个调用点**（m2 的 bench、m5 的
两臂）。可操作的形式是一句话：**对每一个"比较两个东西"的检查，先问"什么故障是两边
共有的"**；只要答案不是"没有"，那个故障就需要在每一边各设一道绝对门槛，而不是把
两边比一比。

**§5.2 —— 持有一个 allocation 不等于持有这台机器。** 本集群的 Slurm 会把
**两个活的 allocation 放在同一台主机上**，而另一个持有者的作业用其它任何手段都
和"残留物"无法区分：`docker ps`、`rocm-smi`、`docker inspect .Created`、对**自己**
作业的 `scontrol`，四者互相印证却都看不见对方。**唯一能回答"还有谁持有这台机器"的
工具是 `squeue -w <node>`**，杀任何东西之前先跑它、逐行读完。这条是因为真的误杀了
同事一个在跑的作业才写下来的。同一天的镜像错误是反过来的：用 `squeue` 去问**硬件**
是否空闲——它答不了。合起来是一句口诀：**要知道卡是否空闲就看卡，要知道卡是谁的就
看队列。**

§5.2 同时把删除规则收紧了：**路径里不含 `yihou` 或 `/tmp` 的，一律不删**；不再局限
于 `/shared_nfs`，且**跟随 mount**（容器里删的就是宿主机的文件）；并且**不是一条可
以权衡的启发式**。写入侧的对应原则是 m4 发现的：`pip install -e` 会往源码树里写
`*.egg-info`，所以**你不会去删的地方，也不要去写**——先拷到本地 scratch（62 MB 用了
12 秒）。

## 1. Handoff 种类

一共 **15 种**，全部在 `main.yaml` 的 `handoffs:` 里列出（权限自上而下继承，未列
在那里的 kind 任何子图都无法获得授权），每一种的定义写在它的生产者所在的
`steps/*.yaml` 里。

命名规则是 `${mode}.${result_type}`（CONTRACT.md §1.1）：点号左边是**配置**，不是
角色名——所以没有 `baseline` / `profiled` 这种叫法。

| # | kind | content_type | 生产者（leaf） | 消费者（leaf） | 携带什么 |
|---|---|---|---|---|---|
| 1 | `deploy_kit` | `code` | m1 `deploy_and_prove` | m2 `run_profiling_mode_off`、`run_profiling_mode_on`；m3 `identify`、`build_workset`；m4 `optimize_kernel`；m5 `apply_patch`、`integrate_and_verify`、`packup` | 一次被证明可用的部署，打成可直接跑的 kit；`codes/environment.yaml` 是全流程的 environment 记录 |
| 2 | `profiling_mode_off.bench_result` | `reproducible` | m2 `run_profiling_mode_off` | m2 `merge_profiling_evidence` | profiler 脱离、CUDA graph 打开时的吞吐与时延——这组数字才有意义 |
| 3 | `profiling_mode_on.bench_result` | `reproducible` | m2 `run_profiling_mode_on` | m2 `merge_profiling_evidence` | 同一负载在 profiler 挂载、CUDA graph 关闭下的结果，用来给 trace 做已知负载参照 |
| 4 | `profiling_mode_on.profile_result` | `reproducible` | m2 `run_profiling_mode_on` | m2 `merge_profiling_evidence` | 该窗口的 torch profiler trace，能被官方工具 load |
| 5 | `profiling_mode_on.kernel_table` | `structured_text` | m2 `run_profiling_mode_on` | m2 `merge_profiling_evidence` | Magpie 的 per-kernel 排名（含 `table.csv`） |
| 6 | `profiling_evidence` | `reproducible` | m2 `merge_profiling_evidence` | m3 `rank`、`build_workset`；m5 `integrate_and_verify`、`packup` | 阶段 2 唯一的对外出口：两条 bench + trace + kernel 排名，共用一份 environment（M2.9） |
| 7 | `kernel_worklist` | `structured_text` | m3 `rank` | m3 `identify` | 排序并分桶后的候选清单，以及被排除者的理由 |
| 8 | `operator_identity` | `structured_text` | m3 `identify` | m3 `build_workset` | 每个候选从 device symbol 解析到源码位置与 launcher，并记录解析等级 |
| 9 | `operator_workset` | `code` | m3 `build_workset` | m4 `optimize_kernel`；m5 `apply_patch`、`packup` | 每个候选算子一套可运行的优化 workset：定义、shapes、正确性测试、性能测试（M3.7 合并了原阶段 3/4 两个 kind） |
| 10 | `kernel_optimization` | `code` | m4 `optimize_kernel` | m5 `apply_patch`、`integrate_and_verify`、`packup` | 一个被 workset 自己的工具验证过的优化 kernel，外加如何 apply 回引擎 |
| 11 | `patch_overlay` | `reproducible` | m5 `apply_patch` | m5 `integrate_and_verify`、`packup` | 逐文件的 bind-mount 计划，每个文件带 before/after hash |
| 12 | `stock.measurement` | `reproducible` | m5 `integrate_and_verify` | m5 `packup` | 未打补丁那一臂：部署证据、正确性结果、replay 轮次 |
| 13 | `patched.measurement` | `reproducible` | m5 `integrate_and_verify` | m5 `packup` | 打了补丁那一臂：同样的测量，外加"补丁确实在跑"的证据 |
| 14 | `integration_report` | `structured_text` | m5 `integrate_and_verify` | m5 `packup` | 两臂并排，以及接受或拒绝这次改动的论证（含 `report.md`） |
| 15 | `e2e_packup` | `code` | m5 `packup` | 无（`main.task.outputs`，整个 flow 唯一的对外产物） | 全流程打成一份复现 kit：各阶段的 results / logs / scripts |

两点容易看错的地方：

- **`deploy_kit` 之所以要一路传到 m5**，不只是因为它带 environment 记录，还因为
  m3、m4 必须**在 m1 拉起来的那个容器里跑东西**（workset 的测试、KernelForge）。
- 全流程**没有任何 `serve_*` task、也没有 `deployment_*` handoff**。M2.5/M5.2 禁止
  "A 负责部署、B 负责使用"，所以需要服务的 task 自己在 STEPS 里拉起并拆掉。相对
  五个 demo，这一条删掉了 5 个 task 和 8 个 handoff kind。

## 2. Validators

**21 个，全部是 program validator。** `validator.schema.json` 有 `agent` 字段，
但本包 21 个定义**没有任何一个声明 `agent`**，因此都走各自 `assets/*.validator/check.py`
的程序体——这正是 mission G4.1「能程序化的，尽量不用 ai」。

**21 个的 `strength` 全是 `strong`**（strong = 失败即停图）。没有 weak validator，
这是逐条读 `steps/*.yaml` 数出来的，不是从名字猜的。

三个共享 validator 定义在 `steps/common.yaml`，由 leader 持有；其余按阶段归属。

| # | validator | 定义处 | 评判的 kind | dimension | cost | 实际检查什么 |
|---|---|---|---|---|---|---|
| 1 | `check_environment` | common | **全部 15 种** | completeness | seconds | handoff 带的 `environment.yaml` 能通过包内 schema，`fixed` 七个字段齐全；跨输入比对 node/gpu_arch/image_id/model_path，`runtime` 只报告不判定。另有 `invariants`（`devices_within_count`）表达**字段之间的关系**——JSON Schema 只能把值和常量比，用兄弟字段的值做上界需要 draft 2020-12 没有的 `$data` |
| 2 | `check_kernel_table` | common | `profiling_mode_on.kernel_table` | usability | seconds | Magpie 六个列名、行数下限、`% Total` 求和落在 [80, 100.5]、top-N 占比 ≥50%、有 `Input Shapes`、head 里至少 10 行能指出 Python launcher（M3.5：m2 与 m3 共用同一份定义） |
| 3 | `check_command_parses` | common | 7 个 `reproducible` kind | usability | seconds | `items/command`（或 `script`）在**它自己 shebang 指定的那个 shell** 下能 parse，且可执行。cheat_for_mock 里 14 个 command 有 11 个不过 |
| 4 | `check_deploy_kit` | m1 | `deploy_kit` | completeness | seconds | 规则全部读自 `assets/schemas/deploy_kit.layout.yaml`，body 自己不带规则；含 `served_name_not_a_path`、`mode_readback`、`completion_evidence` 等 |
| 5 | `check_deploy_serves` | m1 | `deploy_kit` | usability | **gpu_hours** | 用 kit 自己的 `scripts/deploy.sh` 真拉起服务，跑 `probes.yaml` 探针，再压 1k in/1k out、并发 16、三分钟；无论成败都执行 teardown |
| 6 | `check_bench_result` | m2 | `profiling_mode_off.bench_result`、`profiling_mode_on.bench_result` | completeness | seconds | 六条规则：五份导出齐全、JSON 过 `bench_result` schema、请求数 ≥50（`--var min_requests`）、四个指标存在、逐请求算出的错误率 ≤5%、必须是 streaming。**外加 CONTRACT §4.6 的 graph ceiling 绝对门槛**（`assets/lib/graph_ceiling.py`）；`profiling_mode_on` 按设计关闭 CUDA graph，所以在那条线上返回 `None`（记为 note，不判错） |
| 7 | `check_trace_coverage` | m2 | `profiling_mode_on.profile_result` | usability | seconds | **自己打开 trace 数内容**（登录节点上的流式 CTF reader，不依赖 torch）：每个 TP rank 一份可读 trace、每 rank GPU kernel ≥1000、时间跨度合理、stack 窗口的 Python 函数数 |
| 8 | `check_profiling_evidence` | m2 | `profiling_evidence` | completeness | seconds | 五条**跨部件**规则：四个 part 都在且 `parts.json` 不多不少；两条线是同一次部署（比 node 与 image digest，**不比容器**）；两条 bench 的负载配置一致 |
| 9 | `check_worklist_shape` | m3 | `kernel_worklist` | completeness | seconds | 过 `kernel_worklist` schema；随件 schema 与包内逐字节相同；分桶占比求和 ∈[80,100.5] 且桶内计数自洽；每条排除都写了理由 |
| 10 | `check_identity_resolved` | m3 | `operator_identity` | trustworthiness | seconds | 每个候选带 resolution level，未解析的要写 `excluded_reason`/`resolution_hint`。`min_resolve_ratio: 0.0` 是**故意的**：C++ 派发的 Tensile GEMM 本来就没有 Python 帧，设下限只会逼出编造的 `source_file_path` |
| 11 | `check_workset_shape` | m3 | `operator_workset` | completeness | seconds | 过 `workset.schema.json`；≥3 个 shape 且 ≥3 个带性能角色；correctness/performance 两个入口存在、非空、可执行；harness 副本与包内 `harness/` 逐字节相同 |
| 12 | `check_workset_runs` | m3 | `operator_workset` | trustworthiness | **gpu_hours** | **本包信任链的根**：真的在本硬件上重跑 workset 自己的入口（≥1 个 shape，5 组 ×10 次，rsd ≤0.10），并与 workset 印出的数字对账。M4.3.5 让 m4 直接采信 workset 的 baseline，安全性完全建立在这一条上。它要上节点，而 **validator 不声明 agent、启动环境是封闭的**，所以 `transport_path`/`transport_env`/`measure_gpu` 三个 arg 用的是 m1 的拼写与默认值，一个 `--var` 驱动所有要上节点的 validator |
| 13 | `check_optimization_shape` | m4 | `kernel_optimization` | completeness | seconds | m4 输出门的**便宜那一半**：文档过 `kernel_optimization` schema；随件 workset 快照有效；`apply.integration_point` 与 workset 声明的 `edit_target` 逐字段一致；四份文档的内容行下限、REPRODUCE.md 里 ≥4 条命令、六个证据文件存在 |
| 14 | `check_speedup_substantiated` | m4 | `kernel_optimization` | trustworthiness | **gpu_hours** | 昂贵那一半，且是对上一版规则的**反转**（M4.3.5）：不再自己重测 baseline，而是证明"用的就是 workset 的 baseline，且前提成立"。`gpu_arch/shapes/dtype/operator` 不符→abort；`rocm/image_id/torch/driver` 不符→warning 但必须记进 `run_environment.warnings[]`；正确性必须全过。同样加了 `transport_path`/`transport_env`/`measure_gpu`——**在此之前它的重测根本离不开登录节点，所以这个 validator 从未评判过一个真实的 kernel** |
| 15 | `check_overlay_applies` | m5 | `patch_overlay` | completeness | seconds | mount 计划成形、文件在 handoff 里、hash 与计划相符，**并且与被替换的文件不同**（最后这条最关键：能干净打上却什么都没改，会让两臂字节相同、下游全部因为错误的理由通过）；Python 文件额外 `compile()` |
| 16 | `check_patch_live` | m5 | `patched.measurement` | trustworthiness | seconds | 补丁**真的在跑**：`docker inspect` 看到只读 mount；在运行中的容器里重新 hash 出 `sha256_patched`；**运行期 marker 自 2026-09-04 起默认必须有**（`require_runtime_marker` 由 `false` 改为 `${require_runtime_marker:-true}`）。翻转它的实验：一个 in-container hash 完全正确、文件里塞了 2 ms sleep、`.pyc` 当分钟编译的 overlay，测出来**和 stock 一模一样**——静态证据完美，却分不清"挂上了但没执行"和"执行了但没效果"，而这正是这个 validator 存在的唯一理由。`__file__` 证明不了任何事——bind mount 不改容器内路径 |
| 17 | `check_measurement_order` | m5 | `stock.measurement`、`patched.measurement` | trustworthiness | seconds | 两臂跑了同样的步骤、同样的顺序（`smoke, needle, probe, lm_eval, bench`）、时间上不重叠、patched 在 stock 结束之后才开始；并用 AIPerf 自己的 `start_time`/`end_time` 交叉印证——因为 `steps.json` 是被测者自己写的 |
| 18 | `check_acceptance` | m5 | `stock.measurement`、`patched.measurement` | completeness | seconds | 三类正确性证据都到齐且各自可读：固化的四个 smoke check、≥3 个临时生成的 case（M5.4 防作弊）、每个 eval ≥20 题被判分、needle 的实际 prompt token 数达到声称的 95%。**它不比较两臂** |
| 19 | `check_bench_report` | m5 | `stock.measurement`、`patched.measurement` | completeness | seconds | 每一轮 replay 都真的发了流量并写了机器可读报告；与 m2 **共用** `bench_result` schema（G2）；请求数 ≥50（`--var **integration_min_requests**`）、轮数对得上、错误率 ≤5%。这个 `--var` 名字是 2026-09-04 拆出来的：此前它和 m2 的 `check_bench_result` **共用 `min_requests`，一个 `--var` 同时移动两个 owner 的判分线**，而任何一方从自己的文件里都看不出来。默认值仍是 50 且**故意重复**——共享默认值是对的，共享 override 不是 |
| 20 | `check_no_regression` | m5 | `integration_report` | trustworthiness | seconds | **不读 `verdict` 字段**，而是从原始数字重算整个接受/拒绝论证，与报告的结论不符即失败（哪怕自己算出来是"接受"）。吞吐回退 ≤5%、TTFT 回退 ≤10%；stock 臂必须复现 m2 的 `profiling_mode_off`（容差 `--var stock_vs_m2_tolerance`，**这条检查本身不可关闭**）；与 m4 单核加速比的对账只报 warning |
| 21 | `check_packup_shape` | m5 | `e2e_packup` | usability | seconds | 四份文档 + 三个目录存在，且按**内容行**（去掉空行、标题、代码围栏）计量，不是只看文件在不在；REPRODUCE.md 按命令行数计量；`results/` 至少 4 个文件。目录也**按实质计量**（`min_dir_files`，递归找非空文件）而不是 `is_dir()`——m2 实测过 `logs/` 只剩一个空子目录、`scripts/` 只剩一个零字节 `run.sh`，在原来的"存在即通过"下都判为完整 |

每个 kind 挂哪些 validator 写在它自己的定义里；`check_environment` 是唯一挂满 15
种的。一个 phase 内部按 `tags.cost` 从便宜到贵排序执行，所以形状类的错误会在几秒
内失败，而不是等到花掉 GPU 小时之后。

## 3. 每个阶段的 task 划分

11 个 leaf、5 个 stage 非叶节点、加 `main` 共 17 个 closure。非叶节点不声明 agent
（closure spec §2.6：工作即子图，二者互斥），也不声明 `resources`（只有 leaf 能
持锁）。

### m1_deploy — 1 个 leaf

| leaf | agent | in | out | 做什么 |
|---|---|---|---|---|
| `deploy_and_prove` | `${m1_agent:-e2e_deployer}`（**AI**） | 无 | `deploy_kit` | 把一句部署指令变成一次被证明可用的 bring-up 和一个可交付 kit。判断工作在于：在共享节点上挑空闲端口、读仓库自己的例子、看着日志连报四分钟 `Health check failed` 还能判断那是 JIT 编译而不是挂死 |

`--var m1_agent=runner` 会换成共享的 program agent，这就是 mock 的开关：`kind: ai`
的 task 不跑 `entry.sh`，所以不换 agent 就没有 mock 路径。m2–m5 四个 AI leaf 同理
（`m3_agent` / `m4_agent` / `m5_agent`）。

### m2_profiling — 3 个 leaf

| leaf | agent | in | out | 做什么 |
|---|---|---|---|---|
| `run_profiling_mode_off` | `runner`（program） | `deploy_kit` | `profiling_mode_off.bench_result` | CUDA graph 打开、不挂 profiler，自己拉起服务→压负载→拆掉 |
| `run_profiling_mode_on` | `runner`（program） | `deploy_kit` | `profiling_mode_on.bench_result`、`.profile_result`、`.kernel_table` | CUDA graph 关闭、挂 profiler 跑同一负载，切窗口、出 Magpie 排名，再拆掉 |
| `merge_profiling_evidence` | `runner`（program） | 上面 4 个 | `profiling_evidence` | 把两条线折成阶段 2 唯一的出口 |

两条线**不共享任何 handoff**（M2.6），所以推导不出数据边；各自都
`resources: {gpu: 8}`，硬件上因此被串行化，"第二条为什么要等"被如实记录成
`WAITING_RESOURCE` 而不是 `WAITING_HANDOFF`。

**但 2026-09-04 补上了一条纯排序边**：`run_profiling_mode_on` 的 `froms` 现在写着
`[run_profiling_mode_off]`（T38）。原因是那个串行化只是**算术的副作用**——两个各要
8 卡的 task 在 8 卡机器上装不下——而这个保证没人写下来过。两条线用不同的端口段和 run
tag，真要并发起来**两边都会正常起来、什么都不会失败**，只是
`profiling_mode_off`（唯一值得引用的吞吐、也是 m5 stock 臂必须复现的那个数）会在同一
批卡上和一个挂着 profiler 的负载并排测。而 `gpu: 8` 的声明本身已经和实践不符了（真实
运行一律 `tp=4`），所以是**这个过度声明在保护测量**，任何人把它"改对"都会移除保护且
不会得到任何报错。`show` 会把这条边报成"admitted with reservations"，因为它分不清
"无 handoff 支撑的依赖边"和"handoff 被删后遗留的边"——这正是 `froms` 存在的意义
（`graph.py:358-371`），所以那条 reservation 是预期输出。

**三个 leaf 的 `agent: runner` 是字面量，不要改成 `${m2_agent:-<某个 ai>}`。**
m1/m3/m4/m5 都写成 `${mN_agent:-…}`，所以 m2 看起来不合群；它不是，它是唯一没有
`kind: ai` body 的 stage，这个不对称就是设计。改了要付的代价：`kind: ai` 的 task 不跑
`entry.sh`，而 `assets/load/line.sh` 有 402 行、**16 个 exit 点和 7 个 `:?` 守卫**，
其中 `:192` 会在"kit 记录的节点不是我们所在的节点"时拒绝。`env:` 块搬不动它们——
`agent/runner.py:801` 写得很直白：*一个环境变量无法指挥一个 agent；对话不是一个读
`os.environ` 的进程*。`env:` 只让值可达，只有 brief 让值被用上。所以那些拒绝不会退化，
它们会**消失**，而第一个症状是一个看起来合理的数字，测自一个没人检查过的部署。

### m3_analysis — 3 个 leaf

| leaf | agent | in | out | 做什么 |
|---|---|---|---|---|
| `rank` | `runner`（program） | `profiling_evidence` | `kernel_worklist` | 对 profile 出来的 kernel 排序并分桶成候选清单 |
| `identify` | `runner`（program） | `kernel_worklist`、`deploy_kit` | `operator_identity` | 把每个候选从 device symbol 解析到源码位置和 launcher |
| `build_workset` | `${m3_agent:-workset_builder}`（**AI**） | `operator_identity`、`profiling_evidence`、`deploy_kit` | `operator_workset` | 每个候选算子建一套可运行 workset，**并且当场测量**（CONTRACT §4.0，没有单独的 `verify_workset`） |

`rank` 和 `identify` 保持 program：它们手上的证据是一个符号名和一张数字表，规则表
加一个 out-of-process finder 比模型更合适（M3.8 提出的疑问记在 `todo.md` T4）。
`build_workset` 用 AI 只为两个判断：哪个现有函数是正确性参照，哪个调用是要被比速度
的在位实现。其余（scaffold、KernelForge 导出、两个入口脚本）都是生成或逐字节复制
的——**agent 不得改动那两个入口**，否则就是自己写自己的尺子。

### m4_kernel_opt — 1 个 leaf

| leaf | agent | in | out | 做什么 |
|---|---|---|---|---|
| `optimize_kernel` | `${m4_agent:-e2e_kernel_optimizer}`（**AI**） | `operator_workset`、`deploy_kit` | `kernel_optimization` | 跑 workset 自带的 forge 一行命令，再跑 workset 自己的正确性测试、然后才是性能测试，最后写出 m5 消费的 handoff |

全流程最小的一个 stage，这是 mission 的设计而非遗漏：`publish_workset` 被删（M4.2），
独立的 `workset` kind 被并入 m3（M4.1/M3.7），m4 也不声明 workset 的 validator，
而是引用 kind 直接继承（M4.4）。readme 的 STEPS 把七步都写成带验收标准的脚本
（`10_read_inputs.py` … `70_selfcheck.sh`），AI 只负责按序执行和读结果——它**没有
任何环节能选择分母**。

### m5_integration — 3 个 leaf

| leaf | agent | in | out | 做什么 |
|---|---|---|---|---|
| `apply_patch` | `runner`（program） | `kernel_optimization`、`operator_workset`、`deploy_kit` | `patch_overlay` | 把 m4 的 kernel 变成逐文件 overlay 计划，并证明每个文件都落得下去 |
| `integrate_and_verify` | `${m5_agent:-e2e_integrator}`（**AI**） | `patch_overlay`、`profiling_evidence`、`kernel_optimization`、`deploy_kit` | `stock.measurement`、`patched.measurement`、`integration_report` | 在一个 session、一个节点上依次测完两臂再比较：起 stock→测→拆→起 patched（挂 overlay）→同样地测→拆→对比 |
| `packup` | `runner`（program） | 前八个 handoff 全部 | `e2e_packup` | 把整条流程装成一份复现 kit |

这是整个 refine 里结构变化最大的阶段：`integration-demo` 有 8 个 leaf，这里 3 个。
原来的
`serve_stock → measure_stock → serve_patched → measure_patched → compare`
被 M5.2 压成**一个 task**。代价是：原先三条"承载参数而非数据"的边（真正吃劲的是
`serve_patched ← measure_stock`——patched 的第一件事就是拆掉 stock 部署）不再是图上
的边，而变成一个 readme 里的编号列表。`check_measurement_order` 就是用来让这个差别
可见的：body 把每一步打上时间戳写进 `env/steps.json`，validator 拒绝时间重叠或步骤
序列不一致的一对臂。

`apply_patch` 能保持 program，是因为它读的是 m4 的 `kernel_optimization`，而后者的
`apply` 块是照着 workset 声明的 integration point 写的（M5.1.1），并已被
`check_optimization_shape` 逐字段核对过。

**m5 是"全流程共用一个 docker container"这条规则的既定例外**（CONTRACT §5，
mission G5.1 授权）：容器一生只持有一个状态，这正是两臂设计存在的理由。m1–m4 共用
一个，m5 用同一个 image、同一份 `environment` 记录起自己的两臂。

## 4. 两个调试工具

### `assets/lib/replay_root.py` —— 跳级调试

把一次已完成运行里已封存的 handoff 物化成一个 `mock_root`，于是调试运行可以跳过已经
证明过的 stage、把其余的 mock 掉：

```
replay_root.py --out <dir> --run <run> [--run <run> ...] [--kind K ...]
# 然后
--var mock_root=<dir> --var mock_stages=m1,m2 --var m1_agent=runner ...
```

> **这是调试加速器，绝不是验收路径。** 用户说得很明确：*最终验收仍然需要一次完整的
> 真实 e2e*。一次跳过了若干 stage 的绿色运行，只证明了它**跑过**的那些 stage；被
> replay 的那些是在它们产出的当天被证明的，仅此而已。**如果你正在读这段是因为你动
> 了"就用这次跳级运行验收吧"的念头——这句话就是为了拦住这个念头而写的。**

它之所以很小，是因为不引入任何新的注入机制：`mock_root` 本来就是 `--var`，mock leaf
本来就把 `<stage>/<kind>/content/` 拷进 `$AGENT_SYS_OUTPUT_<KIND>`。这个工具只是把同
一套机制指向**上一次好的真实运行**而不是 2026-09-02 的封存语料。

两个设计要点值得记：

- **"稳定"说的是 verdict，不是退出码。** 一次跑完了的运行不等于一次通过了的运行，两者
  在 2026-09-04 分了家：rung 1 **封存**了 `deploy_kit`，随后一个 validator 因为一个数
  字拒绝了它。所以稳定性从 `handoffs/<id>/v<N>/validation.yaml` 算——**整个 run tree
  里只有这个文件保留了 validator 的名字**。一个 kind 在阈值 N 上"稳定"，要求 N 次不同
  运行都用**同一组** validator 产出它，且每个 `result: true`。**validator 组不同不算
  低计数的稳定**，它意味着这些产物根本不是被同一把尺子量的，所以报告时带上分布，从不
  平均掉。
- **曾被当成"免费安全网"的那条，实测不成立。** 这里一度写着
  `check_environment` 的 `compare_fixed_across_inputs` 会让"来自另一台节点的注入
  handoff"当场被拒。**它不会。** m5 于 2026-09-04 晚实测（`270710f`、`5458dfd`）：
  每个下游 handoff 都用 `env_render --inherit <被 replay 的 kit>` 渲染自己的记录，
  所以 `node`/`gpu_arch`/`image_id`/`model_path` 四个字段**全是从 replay 里拷来的**、
  彼此一致、比对通过——在一台这次运行没有使用的节点上。又一个 §0 §4.6：**比较看不见
  所有参与方共有的故障。** 真正强制同节点的 `_agree_or_die` 只在**真实执行**的 stage
  里生效，而跳级恰恰把前面那些变成了 mock。现在的守卫是 `replay_root.py` 自己的
  `--node`（必填，不符 rc=2），且只在**构建 root 时**生效。详见
  [`SKIP-AHEAD.md`](SKIP-AHEAD.md) §6.1。
- 它同样盖不住**失效的活资源**（记录里的节点镜像全对、容器却已不存在），也盖不住
  **引擎配置差异**——这就是 §0 里那个 4.5 倍，四个被比对的字段全都相同。

### `assets/lib/run_with_long_stall.py` —— 抬高 stall 门槛

`agent_sys/cli/main.py:911` 把 `stall_after` 默认成 **20 秒**，而这个值在 CLI 上**没有
暴露成开关**，所以只能用一个 launcher 包一层（`agent_sys/cli/` 不在本次活动范围内，
两个半边都已记在 `temp/bugs/`）。

**机制要说准**（这份 docstring 自己被 m2 更正过一次）：一个长的 program stage **是**
会被算作 `holding` 的——`holding` 的判据是 `_is_running` 且没有停在 escalation 上，
"处于模型调用中"只是注释里的举例，不是谓词。因此**杀掉运行还需要 `blocked` 非空，而
那只能由一次 escalation 产生**。抬高 `stall_after` **不修复任何结构性阻塞，也不假装
修复**：它买的是余量——让一次真的发生了的 escalation 不至于在某个真实 stage 跑到一半
时把运行撕掉，让失败变得可读而不是瞬间毙掉。rung 2b 的实测就是这样死的：
`deploy_and_prove` 已经成功、3 个 strong verdict、`deploy_kit` 有效，
`profiling_mode_off.bench_result` 正在 GENERATING，然后"20 秒无变化"，运行结束。
m2 的两条 profiling 线因为 8 卡资源互斥，会产生 **8–10 分钟**的安静窗口，从 rung 2 起
每次都有。

**它交易掉了什么，明说：** 一个真的挂死的运行，现在要等 `--stall-after` 秒才被判定为
stall，而不是 20 秒。这就是被交易掉的全部安全属性，`--timeout`（默认 4 小时）仍然封顶。

## 5. 阶梯位置（诚实版）

promotion 阶梯定义在 `RUN-PLAN.md`：rung 0 全 mock，rung N 表示 stage 1..N 变真。
**每一级是一次独立的 `agent-sys run`，不跳级。**

| rung | `--var mock_stages=` | 变真的 stage | 状态 |
|---|---|---|---|
| 0 | `all` | 无 | 走过，但 rung 0 **在登录节点上不可能完成**，见下 |
| 1 | `m2,m3,m4,m5` | m1 | **已到达**。2026-09-04 在节点 249（TP-1 bring-up）三个 validator 全绿 |
| 2 | `m3,m4,m5` | +m2 | 未完成（rung 2b 被 stall detector 撕掉，见 §4） |
| 3 | `m4,m5` | +m3 | **不存在任何 rung-3 运行** |
| 4 | `m5` | +m4 | 未运行 |
| 5 | `none` | +m5 | 未运行 |

三件必须说清楚的事：

**rung 0 在登录节点上跑不完，这是设计而非缺陷。** mock 图会一路干净地走到
`build_workset` 然后停下：mock 唯一供不出来的是 `evidence/`，因为 evidence 是一次
**测量**。一个伪造 `evidence/` 的 mock 正是 MOCK-MAP 明令禁止的，而且会直接击穿
`check_workset_runs` 的全部意义。所以 mock 停在测量开始的地方，那是正确的停处。
推论是：**"mock e2e green" 从来不等于"不需要硬件"**，它等于"不调模型、不做 bring-up、
不跑 campaign"。stage 3 往后仍然要卡。

**mock 运行的退出码是 5，而且 5 是正确输出。** 封存语料里的 `integration_report` 带的
是一个**被拒绝**的 verdict，而 `check_no_regression` 不采信 `verdict` 字段、自己从原始
数字重算，独立得到 `REJECTED`。于是一个 handoff 封为 `invalid`，完成规则给出 5。把它
变成 0 的三条路都比 5 更糟：换一个能过的 fixture（不存在，因为 validator 会重算，"能过"
就意味着改掉没人挑选过的数字）；放宽门槛（试过一次，`DELIVERY-NOTE-FROM-LEADER.md`
明确说那是错的答案）；声明为 expected failure（框架真有这个功能，而且是个陷阱——声明
**一个** promise 会把**全部 15 个 handoff** 的完成性检查一起关掉）。**所以退出码不承载
结论**，要读的是文件。

**每一级绿了各自不代表什么**：rung 1 绿只代表引擎答对了 11 个探针、一次负载过了门槛——
一个模型、一个节点、一次。而且 rung 1 在 249 上三个 validator 全绿的**同时有四个
`--var` 是失效的**（`kind: ai` agent 一个 `E2E_*` 都没收到，agent 从封存 kit 的默认值
里把参数"找"了回来）——对的答案，错的机制。rung 5 绿也不代表这次优化是好的：两臂差异
超标时，结论仍可能是关于节点而不是关于补丁，那是 `todo.md` **T7**，尚未实现。
**不要放宽 5% / 10% 的门槛。**

## 6. 开放问题与未解分歧

`../todo.md` 现在有 59 条。这里只列会影响评审判断的几类，**不做粉饰**。

**被用户明确否定、但按指示保留的设计（T5）。** 用户对当前 patch 机制的原话是
*"这里的 patch 机制我都不是很认同，本身就应该是 hack sglang 的 registry 或者 python
的运行"*，同时又说 *但现在就这样吧*。所以 `overlay_files` **保留**，分歧记在 todo 里。
保留的技术理由是：两套机制意味着两套"补丁确实在跑"的证明，而证明是贵的那一半。

**T7 —— bring-up 阶段的可比性门禁，未实现。** 这是 5%/10% 门槛唯一缺的对照。曾经有一
轮把门槛放宽到 35%/30% 去应对一个**跨实例**的假象，那是错的方向。另外 m4 实测：在
MI355X 上把同一件事跑 200 轮，rsd 是 **8.5%** 而不是常说的 ~2%——分布是重尾的
（p90 4.19%、p95 9.15%、p99 51.6%），而"一轮"就是 mission 3.2.7 规定的 5 组 ×10 次协议
下限。**诚实的修法是多采样或取中位数，不是放宽门槛。** 且那次测量是在**共享机箱**上做
的（4-7 号卡被别的租户占到 90%），所以它给出的是上界和分布形状，不是下界。

**T49 —— "结论对、解释错"，一天之内四个 owner 八个实例。** validator 给出的判定全部
正确，只有人读的那句话描述了一个比实际更窄的检查。例如 `run_in_container.sh` 打印
*"m1 的 bring-up 已经被拆掉了"*，而实际情况是**节点根本没连上**——一个传输故障穿着
teardown 的外衣，还点了具体某个人的名。代价是无界的：一个读者查证一次发现是假的，
从此不再相信这个 validator 的**下一条**消息。八个里三个是别人读消息读出来的，没有一个
是作者自己发现的。

**T43 / T54 —— 制造出来的 provenance。** 产物在出处上诚实、在含义上不诚实；
`additionalProperties` 是那个洞，声明 `runtime.replayed_from` 并不能补上。

**T19 / `measure_gpu` —— 一个被绑定的标识符，长期没有变量。** 现状是它**在每一个 rung
（0–5）都必填**，因为 `build_workset/entry.sh:168` 在 **mock 路径上也**调用容器脚本。
`shared.yaml` 与 `m3_analysis.yaml` 的注释一度都写着"body 会回落到 4 号卡"——那句话曾
经是真的，回落被删掉之后注释没有报错，它只是开始撒谎，方向还是"你不需要传这个参数"。
rung 1 就栽在这上面。

**T22 —— `E2E_STAGE` 命名的是每个 stage 各自的事实，而 `--var` 一个运行只有一个值**，
所以任何一次运行里最多只有一个 stage 的 `warnings[].stage` 是真的。

**框架侧的 bug：绕过而非修复，共 12 份记录**（`../temp/bugs/`）。都是先记录、再绕过，
只有证据毫无歧义时才修。其中影响评审的两条：`INVALID` 有两个含义而退出码只读成一个；
声明一个 expected failure 会关掉整个运行的完成性检查。

## 7. 待确认

- **`kernel_table_min_rows` / `kernel_table_min_launchers` 实际由哪张表满足**：默认值
  （20 / 10）可从 `steps/common.yaml` 读到，但 mock 用的是 34 行合成种子、真表是 124
  行，两者同名（见 `cheat_for_mock/README.md`）。要定论需要读一次成功运行的
  `check_kernel_table` findings。
- **m3（部分）、m4、m5 在真实运行中的行为**：§3 里这三节的 in/out 与 agent 归属来自
  yaml 且与 `show` 一致，但"做什么"一列来自 readme 与注释，**尚未被一次真实运行验证**
  ——阶梯只到 rung 1。`check_speedup_substantiated` 更是从未评判过一个真实的 kernel。
- **端口 8101-8103 只能用来"配置"一次 bring-up，不能用来"找到"一次 bring-up。** kit 自
  己挑 base 并把实际绑定写进握手用的 `deployment.json`，那才是唯一权威。rung 1 实际绑
  的是 **8114-8117**。对着一个正在服务的引擎去探 8101-8103 会返回 `000`，而这恰好也是
  一个死掉的引擎返回的东西——leader 在一小时内就踩进了这个坑的读者侧。

