# `e2e-flow` — the five stages, in one graph

Deploy → profile → analyse → optimise one kernel → integrate, as **one
`agent-sys run`**.

The five stages also exist as five separate packages next door
(`../{deploy,profiling,analyze,kernel-opt,integration}-demo/`), each driven to a
real cluster run on 2026-09-02. They are not a flow: **a handoff only travels
inside one run's graph**, so five packages are five runs and nothing chains.
This package is the join the repo-root `mission.md` asks for. The five demos are
kept, untouched, as reference and as a fallback.

## Status — 2026-09-04

The contract is frozen, the graph loads, and the bodies are written. The mock
e2e (rung 0) does **not** yet pass end to end — see 中文报告 §4 below.

```
17 closures (11 leaves + 6 non-leaves) · 15 handoff kinds · 21 validators
```

## Read these, in this order

| file | what it settles |
|---|---|
| [`CONTRACT.md`](CONTRACT.md) | **the frozen cross-module contract** — the fifteen kinds, the naming rule, the environment rule, the schema rule, and what each module deletes |
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
`steps/*.yaml` 与 `assets/*.validator/readme.md`，并与
`python3 -m agent_sys.cli.main show` 的输出交叉核对过（17 closures，与下表一致）。

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
| 1 | `check_environment` | common | **全部 15 种** | completeness | seconds | handoff 带的 `environment.yaml` 能通过包内 schema，`fixed` 七个字段齐全；跨输入比对 node/gpu_arch/image_id/model_path，`runtime` 只报告不判定 |
| 2 | `check_kernel_table` | common | `profiling_mode_on.kernel_table` | usability | seconds | Magpie 六个列名、行数下限、`% Total` 求和落在 [80, 100.5]、top-N 占比 ≥50%、有 `Input Shapes`、head 里至少 10 行能指出 Python launcher（M3.5：m2 与 m3 共用同一份定义） |
| 3 | `check_command_parses` | common | 7 个 `reproducible` kind | usability | seconds | `items/command`（或 `script`）在**它自己 shebang 指定的那个 shell** 下能 parse，且可执行。cheat_for_mock 里 14 个 command 有 11 个不过 |
| 4 | `check_deploy_kit` | m1 | `deploy_kit` | completeness | seconds | 规则全部读自 `assets/schemas/deploy_kit.layout.yaml`，body 自己不带规则；含 `served_name_not_a_path`、`mode_readback`、`completion_evidence` 等 |
| 5 | `check_deploy_serves` | m1 | `deploy_kit` | usability | **gpu_hours** | 用 kit 自己的 `scripts/deploy.sh` 真拉起服务，跑 `probes.yaml` 探针，再压 1k in/1k out、并发 16、三分钟；无论成败都执行 teardown |
| 6 | `check_bench_result` | m2 | `profiling_mode_off.bench_result`、`profiling_mode_on.bench_result` | completeness | seconds | 六条规则：五份导出齐全、JSON 过 `bench_result` schema、请求数 ≥50、四个指标存在、逐请求算出的错误率 ≤5%、必须是 streaming |
| 7 | `check_trace_coverage` | m2 | `profiling_mode_on.profile_result` | usability | seconds | **自己打开 trace 数内容**（登录节点上的流式 CTF reader，不依赖 torch）：每个 TP rank 一份可读 trace、每 rank GPU kernel ≥1000、时间跨度合理、stack 窗口的 Python 函数数 |
| 8 | `check_profiling_evidence` | m2 | `profiling_evidence` | completeness | seconds | 五条**跨部件**规则：四个 part 都在且 `parts.json` 不多不少；两条线是同一次部署（比 node 与 image digest，**不比容器**）；两条 bench 的负载配置一致 |
| 9 | `check_worklist_shape` | m3 | `kernel_worklist` | completeness | seconds | 过 `kernel_worklist` schema；随件 schema 与包内逐字节相同；分桶占比求和 ∈[80,100.5] 且桶内计数自洽；每条排除都写了理由 |
| 10 | `check_identity_resolved` | m3 | `operator_identity` | trustworthiness | seconds | 每个候选带 resolution level，未解析的要写 `excluded_reason`/`resolution_hint`。`min_resolve_ratio: 0.0` 是**故意的**：C++ 派发的 Tensile GEMM 本来就没有 Python 帧，设下限只会逼出编造的 `source_file_path` |
| 11 | `check_workset_shape` | m3 | `operator_workset` | completeness | seconds | 过 `workset.schema.json`；≥3 个 shape 且 ≥3 个带性能角色；correctness/performance 两个入口存在、非空、可执行；harness 副本与包内 `harness/` 逐字节相同 |
| 12 | `check_workset_runs` | m3 | `operator_workset` | trustworthiness | **gpu_hours** | **本包信任链的根**：真的在本硬件上重跑 workset 自己的入口（≥1 个 shape，5 组 ×10 次，rsd ≤0.10），并与 workset 印出的数字对账。M4.3.5 让 m4 直接采信 workset 的 baseline，安全性完全建立在这一条上 |
| 13 | `check_optimization_shape` | m4 | `kernel_optimization` | completeness | seconds | m4 输出门的**便宜那一半**：文档过 `kernel_optimization` schema；随件 workset 快照有效；`apply.integration_point` 与 workset 声明的 `edit_target` 逐字段一致；四份文档的内容行下限、REPRODUCE.md 里 ≥4 条命令、六个证据文件存在 |
| 14 | `check_speedup_substantiated` | m4 | `kernel_optimization` | trustworthiness | **gpu_hours** | 昂贵那一半，且是对上一版规则的**反转**（M4.3.5）：不再自己重测 baseline，而是证明"用的就是 workset 的 baseline，且前提成立"。`gpu_arch/shapes/dtype/operator` 不符→abort；`rocm/image_id/torch/driver` 不符→warning 但必须记进 `run_environment.warnings[]`；正确性必须全过 |
| 15 | `check_overlay_applies` | m5 | `patch_overlay` | completeness | seconds | mount 计划成形、文件在 handoff 里、hash 与计划相符，**并且与被替换的文件不同**（最后这条最关键：能干净打上却什么都没改，会让两臂字节相同、下游全部因为错误的理由通过）；Python 文件额外 `compile()` |
| 16 | `check_patch_live` | m5 | `patched.measurement` | trustworthiness | seconds | 补丁**真的在跑**：`docker inspect` 看到只读 mount；在运行中的容器里重新 hash 出 `sha256_patched`；可选的运行期 marker。`__file__` 证明不了任何事——bind mount 不改容器内路径 |
| 17 | `check_measurement_order` | m5 | `stock.measurement`、`patched.measurement` | trustworthiness | seconds | 两臂跑了同样的步骤、同样的顺序（`smoke, needle, probe, lm_eval, bench`）、时间上不重叠、patched 在 stock 结束之后才开始；并用 AIPerf 自己的 `start_time`/`end_time` 交叉印证——因为 `steps.json` 是被测者自己写的 |
| 18 | `check_acceptance` | m5 | `stock.measurement`、`patched.measurement` | completeness | seconds | 三类正确性证据都到齐且各自可读：固化的四个 smoke check、≥3 个临时生成的 case（M5.4 防作弊）、每个 eval ≥20 题被判分、needle 的实际 prompt token 数达到声称的 95%。**它不比较两臂** |
| 19 | `check_bench_report` | m5 | `stock.measurement`、`patched.measurement` | completeness | seconds | 每一轮 replay 都真的发了流量并写了机器可读报告；与 m2 **共用** `bench_result` schema（G2）；请求数 ≥50、轮数对得上、错误率 ≤5% |
| 20 | `check_no_regression` | m5 | `integration_report` | trustworthiness | seconds | **不读 `verdict` 字段**，而是从原始数字重算整个接受/拒绝论证，与报告的结论不符即失败（哪怕自己算出来是"接受"）。吞吐回退 ≤5%、TTFT 回退 ≤10%；stock 臂必须复现 m2 的 `profiling_mode_off`（容差 `--var stock_vs_m2_tolerance`，**这条检查本身不可关闭**）；与 m4 单核加速比的对账只报 warning |
| 21 | `check_packup_shape` | m5 | `e2e_packup` | usability | seconds | 四份文档 + 三个目录存在，且按**内容行**（去掉空行、标题、代码围栏）计量，不是只看文件在不在；REPRODUCE.md 按命令行数计量；`results/` 至少 4 个文件 |

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

两条线**在图上是并行的**（M2.6：它们不共享 handoff，所以推导不出边），但各自都
`resources: {gpu: 8}`，所以硬件上被串行化。这样"第二条为什么要等"被如实记录成
`WAITING_RESOURCE` 而不是 `WAITING_HANDOFF`。

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

## 4. 当前状态（诚实版）

**mock e2e（rung 0）尚未端到端跑通。**

- m1_deploy、m2_profiling 作为**整个 stage** 已经跑完。
- 运行推进到 m3 的 `build_workset`，`operator_workset` 已封存。
- 随后在 `check_workset_shape` 崩溃：`ModuleNotFoundError: No module named 'referencing'`
  ——validator body 的运行环境缺 `jsonschema` 的依赖，属于环境问题而非 handoff 内容
  问题。
- 因此 m4、m5 在 rung 0 中**尚未被执行过**，本文档中它们的描述来自定义文件，不来自
  一次成功的运行。

不要把本包描述成"已经端到端可用"。

## 5. 需要确认的点（待确认）

- **`kernel_table_min_rows` / `kernel_table_min_launchers` 的实际生效值**：
  `steps/common.yaml` 写成 `'${kernel_table_min_rows:-20}'`、
  `'${kernel_table_min_launchers:-10}'`，默认值可读；但 mock 用的那张表是 34 行的
  合成种子、真表是 124 行（两者同名，见 `cheat_for_mock/README.md`），所以**在
  rung 0 里这两个门槛实际由哪张表来满足，待确认**——需要读一次成功运行的
  `check_kernel_table` findings 才能定论。
- **m4、m5 的 leaf 在真实运行中的行为**：上面 §3 里 m4/m5 两节的 in/out 与 agent
  归属来自 yaml 且与 `show` 一致，但"做什么"一列的描述来自 readme 与注释，**尚未被
  一次成功的运行验证**。

