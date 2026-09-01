# integration-demo 设计方案（rev.2，待 review）

本文用中文写给人 review。落地时 package 内的 `*.yaml` / `readme.md` / 脚本一律用英文，遵循 `../temp/mission.md` 的 RULES 第 6 条。

rev.2 相对 rev.1 的变化，全部来自落地过程中的实测，**每一条都推翻了 rev.1 写定的一个判断**：

1. **needle 的绝对通过门槛被取消**（§5.4.2）。rev.1 写「判定档 76 000 token 实测三个深度全过」，据此设了门禁。九次测量之后确认：结果对长度不单调、对 needle 措辞敏感、对生成预算反向敏感——把预算从 256 提到 2048 会让两个原本通过的深度失败。没有任何配置能稳定过三个深度，门槛无从设起。改为只断言两条事实（长度达标、至少一个深度取回），判定重心移到双臂对比。
2. **`froms` 不参与调度**（§4.1）。rev.1 靠 `serve_patched → measure_stock` 这条纯排序边保证两臂不并发。实测两个 serve 任务被同时派发。改为让这条边带上真实 handoff。框架缺陷记在 `../temp/bugs/004`。
3. **probe 从执行门禁改为结论门禁**（§5.4.3）。rev.1 照搬 1P1D kit 的「probe 挂就不跑 eval」。实测一次 probe 波动让整臂已完成的测量作废。改为 eval 照跑、结论标记为 `uninterpretable`。
4. **容器路径必须用 `@NAME@` 占位符**（§3.2）。发布封印拒绝 manifest 里的 `/sgl-workspace/...`，而展开表本身同样含绝对路径、也不能发布，只能留在包里。
5. **生成预算是一个会伪装成正确性问题的旋钮**（§8）。引擎带推理解析器，答案约占生成 token 的十分之一；按答案长度设预算，测到的是解析器。

对应 `temp/mission.md` 的**接入**阶段：拿到 kernel optimization 的交付物，做端到端正确性测试和性能验证，包含三项——服务启动的简单验证（长短文本 / needle）、llm-eval 验证、端到端性能回归测试。上游是 KernelForge（尚未落地，本包自带 mock），再上游是 `analyze-demo` 的 `operator_workset`。本阶段是流水线的最后一环，其输出即整条链路的验收结论。

文中带「实测」字样的数字都来自 2026-08-31 在 `smci355-ccs-aus-n04-33` 上的手工验证，脚本在 `temp/manual/scripts/`，完整记录在 `temp/manual/FINDINGS.md`。

---

## 1. 结论摘要

流水线由 8 个叶子任务组成，全部是 program task，产出 10 种 handoff、8 个 validator：

```
main（非叶，无 agent）
 ├── seed_patch       → kernel_patch                        froms: []
 ├── apply_patch      → patch_overlay                       froms: [seed_patch]
 ├── serve_stock      → deployment_stock                    froms: [apply_patch]
 ├── measure_stock    → acceptance_stock, bench_stock       froms: [serve_stock]
 ├── serve_patched    → deployment_patched                  froms: [measure_stock]
 ├── measure_patched  → acceptance_patched, bench_patched   froms: [serve_patched]
 ├── compare          → integration_report                  froms: [measure_stock, measure_patched]
 └── packup           → integration_packup                  is_end
       froms: [seed_patch, apply_patch, serve_stock, measure_stock, serve_patched, measure_patched, compare]
```

全部跑在一台 MI355X 上，拓扑与 `profiling-demo` 相同（单节点 MIX + 同机 AIPerf 容器），区别只有两处：本阶段两轮都开 CUDA graph（不采 profile，所以没有理由用不可引用的配置），以及 patched 轮多了一组 bind mount。

已确认的决策：

| 项 | 结论 | 章节 |
|---|---|---|
| stage 4 未落地 | 本包自带 `seed_patch` mock，与 `analyze-demo` 用 `seed_table` mock `kernel_table` 同一手法 | §3.1 |
| 优化 kernel 如何进引擎 | **逐文件 bind mount**，不整树复制、不重建镜像 | §3.3 |
| patch 如何定位 | 镜像引用 + 每个文件的 sha256，不用 git commit | §3.2 |
| 基线从哪来 | 同一场次、同一节点、同一 trace 的 stock 臂，不复用 `profiling-demo` 的历史数字 | §4.1 |
| needle 判据 | **无绝对通过门槛**。只断言长度达标与至少一个深度取回；判定重心是双臂对比。rev.1 的「76 000 token 三个深度全过」结论被实测推翻 | §5.4.2 |
| llm-eval 判据 | Newcombe 区间比较双臂，不比绝对分 | §5.4.3 |
| 性能判据 | 每臂两轮回放（trace-cold / trace-warm），同类相比 | §5.4.4 |
| CUDA graph | 两臂都开 | §4.1 |

---

## 2. 现场核查结果

2026-08-31 在 `n04-33` 上实测，结论直接影响设计。完整记录见 `temp/manual/FINDINGS.md`。

| 项目 | 状态 | 影响 |
|---|---|---|
| `infera/engine-sglang:glm53-flash` 镜像 | **有**，`a9e6029b8750`，`profiling-demo` 构建的那份 | 无需重建，但**镜像是节点本地的**，换节点要重建（实测 9 分 25 秒） |
| AIPerf `nvcr.io/nvidia/ai-dynamo/aiperf:0.12.0` | **有** | |
| GLM-5.3-Flash FP8 权重 | **有**，`/apps/qiongzhu/models/GLM-5.3-Flash-FP8`，306 GB | |
| `sglang.test.run_eval` 在引擎镜像内 | **有**，且接受 `--thinking-mode glm-45` / `--gsm8k-data-path` / `--repeat` / `--num-examples` | llm-eval 不需要往容器里装任何东西 |
| GSM8K test 划分 | **有**，`sglang_1p1d_glm5.2/lm_eval/data/test.jsonl`，1319 行 | 作为包变量传入，下载留作 fallback |
| sglang 安装方式 | **editable**，`Editable project location: /sgl-workspace/sglang/python` | 逐文件 bind mount 直接生效，不需要重装 |
| sglang python 树体量 | 87 MB | |
| aiter 树体量 | **6.9 GB** | 整树复制不可行，这是逐文件挂载的决定性理由 |
| sglang / aiter 是否 git 仓库 | 都是，但 sglang 工作树相对自己的 HEAD 是脏的 | patch 不能用 `git diff`，见 §3.2 |
| needle 脚本 | **仓库里没有**，1P1D README 提到的 238K needle 测试没留下脚本 | 本包从零写 |
| 计算节点外网 | 通 | GSM8K fallback 下载可用 |
| 本地盘 `/data` | 60 T，可用 54 T | 产物落这里 |
| 共享盘 `/apps` | 94% 已用，仅剩 3.7 T | 不写大产物 |

---

## 3. 输入契约：`kernel_patch`

### 3.1 stage 4 尚未落地，所以本包自带 mock

`analyze-demo` 的 DESIGN 把 forge-loop 的**输入**定义得很完整（`--kernel` / `--driver` / `--invocation-spec-file` / `forge_task.yaml` / driver 的 stdout 契约），但对它的**输出**只留下两句话：forge-loop 在 git worktree 里就地编辑并「导出 patch」，以及命令行上有一个 `--result-json <out>/forge_result.json`。`forge_result.json` 的字段、patch 的形态、被替换 kernel 的命名方式，都没有规格。

因此本阶段做两件事：

1. 在 §3.2 把 `kernel_patch` 这个 kind 定义出来，作为**向 stage 4 提出的输入契约**。stage 4 落地时按它产出即可合图。
2. 自带一个 `seed_patch` 叶子产出符合该契约的 mock，让本阶段独立可跑。这与 `analyze-demo` 用 `seed_table` mock `kernel_table` 是同一手法，合图时删掉这个叶子、把 `apply_patch` 的 `froms` 指向 stage 4 的产出任务即可。

**mock 刻意是语义等价的**（只加日志，不改计算）。这不是偷懒，是为了让 validator 自身可被检验：一个语义等价的 patch，正确性差异的期望值是**零**，性能差异的期望值落在噪声内。如果 `compare` 在这种输入下报出回退，那是判据或测量的问题，不是被测对象的问题。mock 的局限也要写清楚：它证明不了「优化后的 kernel 真的更快」，只证明整条接入链路能把一个 patch 送进引擎、跑完三类验收、给出可复核的对比。

### 3.2 `kernel_patch` 携带什么

`content_type: code`，item 为 `codes` 与 `watchout`。选 `code` 而不是 `structured_text`：本 handoff 要携带一个目录（若干 `.patch` 文件加一份 manifest），而 `structured_text` 的可选 item 只有 `text.json` / `text.yaml` / `text.xml` / `schema` 四个，`handoff/content.py:check_items` 会拒绝类型没声明过的顶层 item。

`items/codes/` 的形状：

```
codes/
├── manifest.json
├── patches/
│   └── 0001-<operator_id>.patch      每个被改文件一段标准 unified diff
└── notes/
    ├── forge_result.json             stage 4 的优化报告，mock 里是占位
    └── program.md                    优化说明，来自 workset
```

`manifest.json` 的字段：

| 字段 | 含义 |
|---|---|
| `schema_version` | 本契约的版本，`1` |
| `operator_id` / `logical_operator` | 来自 `operator_identity`，说明这个 patch 优化的是哪个算子 |
| `image` | patch 是对着哪个镜像切的，例如 `infera/engine-sglang:glm53-flash` |
| `apply_mode` | `overlay_files` 或 `rebuild`。本轮只实现前者，后者快速失败并说明原因 |
| `files[]` | 每个被改文件一条：`container_path`、`base_sha256`、`patch`（相对 `patches/` 的文件名）、`change`（`modify` / `add`） |
| `runtime_marker` | 可选。patch 在引擎日志里留下的两个标记：`import` 与 `first_call` 的正则 |
| `expect` | 可选。stage 4 声称的效果：`speedup`、`snr_db`、`baseline_wall_ms`、`optimized_wall_ms` |
| `provenance` | 上游 handoff 的引用：`operator_workset` / `workset_evidence` 的 id |

**为什么用 sha256 而不是 git commit**。sglang 与 aiter 在镜像里都是 git 仓库，但 sglang 的工作树相对自己的 HEAD（`71de97b2…`）是脏的——镜像构建时用 PR #36507 的 overlay 整个替换了 `python/sglang`。对着 HEAD 做 `git diff` 会把这条流水线之外的改动一起带上。所以 patch 是被改文件的普通 unified diff，钉住它的是**文件在镜像里的 sha256** 加镜像引用。`apply_patch` 在 sha256 对不上时拒绝执行，这条检查抓的正是「这个 patch 是对着另一个镜像切的」。

`runtime_marker` 是可选的，但没有它就只剩静态证据。见 §8 第 1 条。

### 3.3 为什么是逐文件 overlay

三个候选：

1. **重建镜像**。最彻底，也是编译型 kernel（HIP / CK / 汇编）唯一的出路。代价是实测 9 分 25 秒一次构建，且每改一次 patch 就要重来。
2. **整树 bind mount**。把 `/sgl-workspace/sglang/python/sglang` 换成宿主机上打过 patch 的副本。sglang 树 87 MB 尚可，但 aiter 树 **6.9 GB**，而 kernel patch 落在 aiter 的概率很高。
3. **逐文件 bind mount**。只把被改的那几个文件挂进去。

选 3。sglang 是 editable 安装，解释器直接读镜像里的树，挂一个文件上去就被编译进去——实测在一个临时容器里挂单文件，marker 在 import 时打出，`__file__` 解析到容器内路径。产物只有几 KB，且挂载清单本身就是可审计的证据。

局限写在 `watchout` 里：只支持修改已有文件与新增文件（新增文件同样逐个挂），不支持删除文件，不支持任何需要编译的改动。`apply_mode: rebuild` 的 patch 在 `apply_patch` 里以一句「本轮不支持，需要重建镜像」失败，而不是假装挂上去了。

**`__file__` 不能作为证据。** 挂载之后容器内路径没变，所以 `m.__file__` 对 stock 与 patched 是同一个字符串。唯一有效的静态证据是**在运行中的容器里重新算这个文件的 sha256**，与 overlay 记录的值比对。这一条决定了 `check_patch_live` 的写法。

---

## 4. agent_sys 图设计

### 4.1 为什么必须双臂同场次

回归测试要有基线。三个来源，只有一个成立：

- **复用 `profiling-demo` 的 `aiperf_baseline`。** 不成立。该包实测同一份配置、同一份 trace 的两次运行相差一个量级：631 tok/s / TTFT 25.9 s（冷）对 1004 tok/s / TTFT 484 ms（复用部署），因为 Mooncake trace 带 `hash_ids`，AIPerf 会展开成真实 token block，前缀命中率直接决定 prefill 的工作量。跨场次的数字不可比。
- **用公开的参考分数。** 对 llm-eval 也不成立。1P1D kit 的 README 说得很直接：绝对分没有外部基线，有意义的比较是与上一次运行的差，或者 `gsm8k` 与 `mixed_prefix_gsm8k` 之间的差。
- **同场次跑一遍未打 patch 的 stock 臂。** 成立，且是唯一成立的。

所以图是两臂：`serve_stock → measure_stock → serve_patched → measure_patched`。两臂之间**只有 bind mount 这一个变量不同**：同一节点、同一镜像、同一权重、同一 trace、同一引擎参数、同样的操作顺序、同样都是冷启动。

`serve_patched` 的 `froms` 指向 `measure_stock` 而不是 `serve_stock`。它的第一步就是把 stock 部署拆掉，所以这条边表达的必须是「stock 臂已经测完」；连到 `serve_stock` 会让 agent_sys 把它和 `measure_stock` 并发调度，在测量中途销毁部署。这条理由与 `profiling-demo` 的 `serve_profiled` 逐字相同。

**但只写 `froms` 不够，这一条是跑出来的。** 2026-08-31 第一次跑全图时，`serve_patched` 与 `serve_stock` 被同时派发，撞在同一个容器名上退出。原因是 `task_graph` 的调度边全部由 handoff 连线推导，一条不共享 handoff 的 `froms` 会在加载期被「admitted with reservations」接受、然后被调度器忽略——而那条警告的原文恰恰写着「it is a dependency that shares no handoff, which is what `froms` is for and is legal」。记在 `../temp/bugs/004-froms-without-a-handoff-does-not-order-execution.md`。

失败是响的，纯属运气：两臂用同一个容器名。若名字不同，两个部署会同时起来抢八张卡，各自在对方运行时完成测量——一个静默错误的对比，正是本阶段存在的意义所要排除的东西。

修法是让这条边带上 handoff：`serve_patched` 消费 `bench_stock`（stock 臂的最后一个产出），并把 stock 臂的步骤序列记进自己的 `env/deployment.json` 的 `preceded_by`。这不是把绕过包装成特性——patched 部署记录本来就该写明它是在 stock 臂测完之后创建的，因为整个对比依赖这一点；消费这个 handoff 只是把这句话从断言变成可核对的事实。

`serve_stock` 的 `froms` 指向 `apply_patch` 而不是空。overlay 建好并校验过只要几秒，而 stock 臂要跑二十多分钟。让 patch 先应用一次，一个挂不上的 patch 在十秒内失败，而不是在半小时之后。

**两臂都开 CUDA graph。** `profiling-demo` 关掉它是因为要采 profile，graph 打开时 profiler 看到的是一次 graph launch 而不是里面的 kernel。本阶段不采 profile，没有理由用一个不可对外引用的配置去做回归。代价是多一次 graph capture，实测 bs≤128 时约 33 秒 / 1.4 GB。

### 4.2 为什么 `measure_*` 是一个任务而不是三个

三类验收（smoke+needle / llm-eval / AIPerf 回放）必须跑在**同一个部署实例**上，且**不能在时间上重叠**——一次饱和 trace 回放期间跑 lm-eval，两边的数字都作废。

如果拆成三个 `froms: [serve_stock]` 的并列兄弟，agent_sys 会并发调度它们，而它们之间没有同步原语。这与 `profiling-demo` 里 `run_profiled` 同时产出 `aiperf_profiled` 和 `torch_trace` 是同一个理由：框架的限制是「每个 subgraph 里每种 kind 只能有一个 producer」，不是「每个 task 只能产一种 kind」。

所以 `measure_<arm>` 是单个 program task，产出两种 handoff（`acceptance_<arm>` 与 `bench_<arm>`），内部时序固定：

```
1. smoke        短提示 / 长生成 / 引擎日志扫描
2. needle       76 000 token 三个深度（判定）+ 127 000 token 三个深度（记录）
3. lm-eval      probe 门禁 → gsm8k → mixed_prefix_gsm8k
4. bench 第 1 轮  AIPerf 回放 trace 前 2 分钟（对 trace 而言是冷的）
5. bench 第 2 轮  同样的回放（对 trace 而言是热的）
```

顺序在两臂之间必须逐字相同，否则第 4 步的「冷」不是同一个冷。这一条写在任务的 readme 里，也写进 `env/measure.json`，`check_bench_report` 会核对两臂记录的顺序一致。

拆成两个 handoff 而不是一个：正确性与性能是两类证据，绑不同的 validator，`compare` 也按两条线分别判定。合成一个会让 `check_acceptance` 不得不同时懂 AIPerf 的 CSV。

### 4.3 十种 handoff kind

| kind | content_type | items | 说明 |
|---|---|---|---|
| `kernel_patch` | `code` | `codes`、`watchout` | stage 4 的交付物，见 §3.2 |
| `patch_overlay` | `reproducible` | `result`、`env`、`command`、`logs`、`watchout` | 挂载清单 `mounts.json` + 打过 patch 的文件副本 + 每个文件的 stock/patched sha256 |
| `deployment_stock` | `reproducible` | 同上五项 | 未打 patch 的 MIX 部署与它服务过的证据 |
| `deployment_patched` | `reproducible` | 同上五项 | 打了 patch 的同一套部署，`env` 里多一份 `docker inspect` 的 Mounts 与容器内实测 sha256 |
| `acceptance_stock` | `reproducible` | 同上五项 | `result/` 下 `smoke.json`、`needle.json`、`lm_eval/`（`.index` + 每个 eval 的 json 与 html） |
| `acceptance_patched` | `reproducible` | 同上五项 | 同 schema |
| `bench_stock` | `reproducible` | 同上五项 | `result/` 下 `round1/`、`round2/` 两个 AIPerf artifact 目录，外加 `summary.json` |
| `bench_patched` | `reproducible` | 同上五项 | 同 schema |
| `integration_report` | `structured_text` | `text.json`、`schema`、`report.md` | 双臂对比与验收结论 |
| `integration_packup` | `code` | `codes`、`watchout` | 按 `experiment-result-packup` 的 `deliverable_layout.md` 组织的交付物 |

`reproducible` 强制要求 `result` 与 `env`，且 `script` / `command` 至少有一个，README 必须含 Purpose / How to run / Result / Environment / Watch out 五节。所有 `reproducible` 的 kind 都带 `command` 而不是 `script`，理由与 `profiling-demo` 相同：`handoff.locality` 拒绝任何命名了固定允许列表之外绝对路径的内容文件，而真实的拉起脚本里有 `/tmp/glm53_mix.log` 这类路径；handoff 携带调用方式，脚本的出处记在 `env/*.json` 里。

`integration_report` 用 `structured_text` 并在 `items_schema.properties` 里显式声明 `report.md`。`check_items` 的规则是 `unknown = set(items) - known - declared`，`declared` 取自 `items_schema` 的 properties——所以只要声明了就不会被拒。`analyze-demo` 的 `kernel_worklist` 声明 `worklist.csv` 是同一个做法。

`integration_packup` 用 `code`：把 packup 塞进 `reproducible` 会把 `results/` 改名成 `items/result`，`REPRODUCE.md` 也没有 item 可当，恰好毁掉 `check_packup_shape` 要检查的东西。

`deployment_stock` / `deployment_patched`、`acceptance_*`、`bench_*` 都是同一套 schema 的两个 kind 名。不共用一个名字，是因为框架限制每个 subgraph 里每种 kind 只能有一个 producer。

各 kind 都不写超出顶层形状的 `items_schema`：产物目录**内部**的结构是 validator 的职责，这是两套机制存在的意义。

### 4.4 八个 validator

`validator.schema.json` 是 `additionalProperties: false` 且没有 `kind` 字段，所以不存在「AI validator」这个 schema 概念；本包沿用 `profiling-demo` 的写法，用 `tags.logic_source` 与 `tags.cost` 表达差别，`cost` 决定同一 phase 内的排序（便宜的先跑）。`validator.inputs` 是多对多绑定，所以两臂可以共用同一个 validator。

| validator | dimension / strength | cost | inputs | 检查内容 |
|---|---|---|---|---|
| `check_patch_shape` | completeness / strong | seconds | `kernel_patch` | `manifest.json` 必需字段齐全；`apply_mode` 在枚举内；每个 `files[]` 条目的 `.patch` 文件存在且能被 `git apply --check` 之外的形式解析（是合法 unified diff，有 `---`/`+++`/`@@`）；`base_sha256` 是 64 位十六进制；`container_path` 是绝对路径且落在已知的容器根下 |
| `check_overlay_applies` | completeness / strong | seconds | `patch_overlay` | `mounts.json` 的每一条都有宿主机文件且非空；`sha256_patched` 与文件实际内容相符；`sha256_stock` 与 manifest 的 `base_sha256` 相符；打过 patch 的文件与 stock 文件确有差异（一个 no-op patch 会让整条流水线变成自己和自己比）；python 文件能通过 `compile()` |
| `check_service_live` | completeness / strong | seconds | `deployment_stock`, `deployment_patched` | 移植自 `profiling-demo`：`/v1/workers` 恰好 1 个且 `disagg_mode == "mixed"`；smoke 的算术答案为 391；worker 日志无 `memory access fault` / `HIP error` / `Traceback`；`env` 里记录的引擎实际 argv 与该 kind 预期的 CUDA graph 设置一致 |
| `check_patch_live` | trustworthiness / strong | seconds | `deployment_patched` | **本包最重要的一个**。`docker inspect` 记录的 Mounts 覆盖了 `mounts.json` 的每一条；在运行中的容器里重算的文件 sha256 等于 `sha256_patched`；若 manifest 声明了 `runtime_marker`，引擎日志里 `import` 与 `first_call` 两个标记都出现过 |
| `check_acceptance` | completeness / strong | minutes | `acceptance_stock`, `acceptance_patched` | smoke 三项都有结论；needle 在判定长度上三个深度全过，且每次请求的 `prompt_tokens` 达到了声明的长度（否则测的不是长上下文）；lm-eval 每个 eval 都有 json 与 html，`.index` 里的 scored 数与 html 里 `Correct Answer` 出现次数一致，且 scored 数不低于下限 |
| `check_bench_report` | completeness / strong | seconds | `bench_stock`, `bench_patched` | 两轮都有四个 AIPerf 必需文件且非空；`request_count` 高于下限；错误率低于 `max_error_rate`；`summary.json` 里 TTFT / TPOT / 吞吐 / 请求数齐全；两臂记录的步骤顺序一致 |
| `check_no_regression` | usability / strong | seconds | `integration_report` | **不读 `verdict` 字段，从原始数字重新算一遍**：Newcombe 区间是否含零、needle 是否有 stock 过而 patched 挂的深度、吞吐与 TTFT 的相对变化是否越过阈值。重算结果与报告自称的结论不一致，直接判 FAIL——一个自称通过的报告不是证据 |
| `check_packup_shape` | completeness / strong | seconds | `integration_packup` | 移植自 `profiling-demo`，按 packup skill 的 `checklist.md` 校验目录形状与内容行数下限 |

`validator/report.py` 的 `blocks_the_task` 规则是：output phase 上一个 handoff 没有绑定任何 validator 会被判为 `unchecked` 并阻塞。上表覆盖了全部十种 kind。

`check_patch_live` 的 dimension 取 `trustworthiness` 而不是 `completeness`，因为它检查的不是「记录是否齐全」，而是「这份记录描述的事情是否真的发生了」。这也是本包唯一一个 `trustworthiness`。

---

## 5. 各任务实现要点

### 5.1 `seed_patch`

产出符合 §3.2 契约的 mock。步骤：

1. 从镜像里 `docker create` + `docker cp` 取出目标文件的原始副本（不启动容器，不占 GPU）。
2. 按 `patch_target` 变量定位目标函数，插入两处语义等价的标记：模块级的 `import` 标记，以及目标函数体首行的一次性 `first_call` 标记（用一个模块级布尔量守住，稳态开销是一次布尔判断）。
3. `diff -u` 产出 patch，`sha256sum` 记录 base。
4. 写 `manifest.json`，`runtime_marker` 填这两个标记的正则。

默认目标是 `sglang/srt/models/glm5_next.py` 的 `Glm5NextDecoderLayer.forward`：实测它没有装饰器，且每个 batch 的每一层都会进。**实测排除过两个看起来更合适的候选**：`swiglu_clamped` 只在 vision 路径上（文本请求永远不进），且带 `@torch.compile`（日志调用会造成 graph break）。

目标是包变量，因为真实的 KernelForge patch 自己会指定，mock 的选择不该看起来像契约。

### 5.2 `apply_patch`

1. 读 `kernel_patch`，`apply_mode != overlay_files` 直接失败并说明原因。
2. 对每个 `files[]` 条目：从镜像取出原始文件，核对 sha256 与 `base_sha256`；不符则失败，报出两个值与镜像引用。
3. 在节点本地 `$work_root/overlay/<run_tag>/` 下打 patch（`patch -p1` 或 `git apply`，两者都不需要目标是 git 仓库）。
4. python 文件跑一次 `compile()`，语法错误在这里失败，而不是在引擎启动 15 分钟后。
5. 写 `mounts.json`：每条 `{container_path, host_path, sha256_stock, sha256_patched, change}`。
6. 把打过 patch 的文件副本、patch 本身与 `mounts.json` 一起放进 handoff——handoff 要自包含，节点回收后仍能复核改了什么。

### 5.3 `serve_stock` / `serve_patched`

两者共用 `assets/serve/round.sh`，靠环境变量分叉，移植自 `profiling-demo/assets/serve/`。相对那份的改动只有两处：

1. **两轮都是 `CUDA_GRAPH=1`**，两轮都不加 router 的 `--enable-profiling`（本阶段不采 profile；少一个参数就少一处两臂之间的差异）。
2. **patched 轮把 `mounts.json` 的每一条翻成一个 `-v host:container:ro`**，追加到 `docker run` 的参数里。stock 轮不追加任何一条。

保留 `reset_gpus.sh` 的硬门禁与幂等 teardown：上一轮残留进程还占着显存时启动 worker，分布式 bootstrap 会以「memory capacity is unbalanced」这种误导性错误退出。

引擎参数沿用 AMD 验证过的配方，两臂逐字相同：

```
--tp-size 8 --trust-remote-code
--dsa-prefill-backend tilelang --dsa-decode-backend tilelang
--kv-cache-dtype bfloat16 --moe-runner-backend triton
--reasoning-parser glm45 --tool-call-parser glm47
--mem-fraction-static 0.85 --context-length 262144 --chunked-prefill-size 8192
--cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled
--cuda-graph-bs-decode 1 2 4 8 16 24 32 48 64 96 128
```

patched 轮额外把容器内实测的文件 sha256、`docker inspect` 的 Mounts、以及引擎日志里 marker 命中的行，写进 `env/patch_live.json`，供 `check_patch_live` 读取。这些证据必须由 serve 任务采集：`check_patch_live` 跑的时候部署可能已经被下一步拆掉了，validator 只能从 handoff 里读。

### 5.4 `measure_stock` / `measure_patched`

共用 `assets/accept/` 与 `assets/bench/` 下的脚本，五个步骤按 §4.2 的固定顺序执行。

#### 5.4.1 smoke（长短文本）

`长短文本` 拆成三个方向，各自能挂而另外两个看不见：

| 方向 | 提示 | 期望 |
|---|---|---|
| 短提示 / 短输出 | `Compute 17 * 23. Reply with only the number.` | `391` |
| 短提示 / 长输出 | 请求 2048 token 的连续描述，**并用 `chat_template_kwargs: {enable_thinking: false}` 关掉推理** | 非空、词数 ≥ 50、无 8-gram 重复超过 4 次 |
| 长提示 / 短输出 | 见 5.4.2 的 needle | 见下 |

外加移植自 `profiling-demo/assets/serve/mix_smoke.sh` 的两段结构证据：`/v1/workers` 恰好一个 `mixed` worker，引擎日志无 fault。

8-gram 重复检查移植自 `probe_accuracy.py` 的软判据，但在这里是**硬判据**：一个吐重复的部署在 lm-eval 上会拿到一个数字而不是一个错误，那个数字与真实回退无法区分。

**长输出这一项单独关掉推理，测了两次才定下来。** 512 预算下 `content` 为空、`finish_reason: length`；提到 2048 后仍然为空——1470 词推理、0 词答案。继续加预算是在和一个「给多少用多少」的模型赛跑。这项检查问的是「这台部署能不能生成长而不退化的连续文本」，推理过程不是它要测的东西，所以用 `enable_thinking: false` 直接要答案：实测 1469 词答案、0 词推理、最大 8-gram 重复 1。needle 与 lm-eval 保持推理开启，因为在那两处推理是被测内容的一部分。

脚本仍保留一条退路：如果部署忽略了这个 kwarg 照样推理，就改判推理文本的退化情况，并在记录里写明判的是哪一段——1470 词不重复的推理同样回答了「能不能生成长而连贯的文本」这个问题，把它记成「没有输出」才是错的。

#### 5.4.2 needle

从零写（仓库里没有），`assets/accept/needle.py`。

- **haystack**：编号递增的 filler 句 `Background fact number <n> is irrelevant to any question.`，保证没有两句相同。
- **needle 是词汇型的**，`The maintenance passphrase for vault <V> is MAGENTA-QUASAR-17.`。数字型 needle 埋在数字型 filler 里，在 76 472 token 的 head 深度上稳定取不回来（三次返回同一个 `'7-3-9-1'`，temperature 0 下确定性复现）。只改 needle 的措辞，同一位置就取回来了——所以那测的是模型能不能分辨两个数字，不是长上下文。
- **三个深度**：0.02 / 0.5 / 0.98。只探一个深度是认证一台坏机器的办法：prefill 按 `--chunked-prefill-size` 分块，一个丢掉除最后一块之外所有块的缺陷，仍然会把 tail 深度的 needle 答对。
- **长度用 `usage.prompt_tokens` 回读校验**，不依赖本地 tokenizer。实测 12.0–12.7 token/句，脚本按这个系数估句数，再断言实测 token 数不低于目标的 95%。
- **`--max-tokens` 默认 2048，且空答案要与取不回来区分开**。引擎带 `--reasoning-parser glm45`，模型先思考、`content` 在思考结束前一直是空的；256 的预算下 31k haystack 的两个深度返回**完全空的 content 加 `finish_reason: length`**，从外面看与「没找到」一模一样。脚本把这种情况标为 `starved` 并单独报错。

**rev.1 写的「判定档实测三个深度全过」这条结论，在实现过程中被推翻了。** 把生成预算从 256 提到 2048 修好了空答案，同时**让检索变差**：79 492 token 三个深度挂两个（middle 返回「I don't have any information about a maintenance passphrase」），而 76 473 token 在 256 预算下三个全过。九次测量放在一起：

| 长度 | needle | 预算 | head | middle | tail |
|---|---|---|---|---|---|
| 31 302 | 数字 | 256 | 过 | 过 | 过 |
| 41 532 | 数字 | 256 | 过 | — | — |
| 51 862 | 数字 | 256 | **挂** | — | — |
| 62 222 | 数字 | 256 | 过 | — | — |
| 76 472 | 数字 | 256 | **挂** | 过 | 过 |
| 76 473 | 词汇 | 256 | 过 | 过 | 过 |
| 127 053 | 词汇 | 256 | **挂** | 过 | 过 |
| 31 082 | 词汇 | 2048 | 过 | **挂** | 过 |
| 79 492 | 词汇 | 2048 | 过 | **挂** | **挂** |

对长度不单调、对 needle 措辞敏感、对生成预算反向敏感——预算小了饿死答案，大了让模型把答案说没。**没有一组配置能稳定地三个深度全过**，所以绝对门槛无从设起。

改成两层：

1. **绝对底线只有两条，都是事实而非行为**：prompt 真的达到了声称的长度（`usage.prompt_tokens` 回读），以及**至少一个深度取回**。后者区分的是「长上下文通路能工作」与「除最后一块 prefill chunk 之外全丢」，正是这个测试被加进来要抓的那个失败。要求三个全过，就是要求一件没有任何配置做到过两次的事。
2. **needle 真正的用途是双臂对比**：stock 取回而 patched 取不回，才算回退。两臂都取不回不是回退，是这台部署在那个长度上的性质。

两档长度仍然保留（默认 76000 与 127000），因为「差异只在某一档出现」本身是信息，但两档都不设通过门槛。

实测每次请求 6–13 秒，整套 needle 一臂不到两分钟。

**本包的 needle 结果不能用来陈述 GLM-5.3-Flash 的长上下文能力。** 上面这张表就是理由。

#### 5.4.3 llm-eval

移植 `sglang_1p1d_glm5.2/engine/lm_eval.sh` 的逻辑，评测器是引擎镜像自带的 `sglang.test.run_eval`，走 OpenAI `/v1/chat/completions`，容器里不装任何东西。

三处必须改：

1. **`probe` 要认 MIX。** 原 `probe_accuracy.py` 硬要求 prefill 与 decode 两种 worker 都注册，而 MIX 下 `/v1/workers` 只有一个 `disagg_mode: mixed`。本包的 `assets/accept/probe.py` 按 `mixed` 判定，其余四项检查（可达性、可答性、重复稳定性、共享前缀不改变答案）逐条保留。

   **probe 门禁的作用点从「执行」改到了「结论」，这一条也是跑出来的。** 原设计照搬 1P1D kit：probe 挂就不跑 eval。2026-08-31 实测时 probe 在 stock 臂挂了一次（`natalia` 单独提问答 72，放在 3300 token 共享前缀之后答 7），于是 eval 被跳过、`.index` 为空、`check_acceptance` 判 invalid，整张图停在 `measure_stock` 的 output phase——一次波动毁掉了两臂中已经完成的那一臂，而原因与被测的 patch 毫无关系。在 1P1D kit 里这个设计是对的：那里有个人盯着，重跑一次十三分钟。这里一轮是一个小时。

   改法：eval 照跑，`probe.ok` 进 handoff，`compare` 在任一臂 probe 挂时把 eval 对比标成 `uninterpretable` 而不是 `same`，并把它列进 `reasons`（所以 patch 不会靠一个读不懂的 eval 通过）。`check_acceptance` 只要求 probe 跑过并留下记录，不要求它通过。既没有东西被默默相信，也没有东西被扔掉。

   附带修的一处：probe 判定答案用「文本里最后一个数字」，所以答错和取错看起来一样。失败时现在把答案文本尾部一并记进 `probe.json`，下一个读的人不必重跑就能分辨。
2. **数据集用包变量指过去**，`gsm8k_data` 默认 `sglang_1p1d_glm5.2/lm_eval/data/test.jsonl`（实测在位，1319 行），`docker cp` 进容器的 `/tmp/gsm8k_test.jsonl`。行数不是 1319 就告警——短文件是下载被截断，会让每次 gsm8k 的分数不可比。
3. **`--thinking-mode glm-45` 必须给。** 引擎带 `--reasoning-parser glm45`，`content` 里只有答案；不给这个参数，一个健康的部署会拿到 0.00。实测该参数在本镜像的 `run_eval` 上可用。

默认跑 `gsm8k` 与 `mixed_prefix_gsm8k` 两个，各 200 题、32 并发。配对是有目的的：两者问同样的题，后者把题放在部分共享的 few-shot 前缀后面，两者之差直接测量前缀复用是否改变答案——而前缀复用被改坏正是 kernel patch 可能造成、其他检查都看不见的失败。

打分题数从 html 报告里数 `Correct Answer` 的出现次数，不用数据集大小：GSM8K 有 1319 行，但 `gsm8k` 打分 1314 题、`mixed_prefix_gsm8k` 打分 1299 题，评测器把 few-shot 例子从评测集里切掉了。区间计算要这个数，而结果 json 不带它。

区间用 Wilson 分数区间（不是 Wald，后者在 p=0 或 1 处宽度为零），双臂之差用 Newcombe 方法。实现落在 `assets/lib/eval_stats.py`，与 `summarise_eval.py` 同一套算法；validator 与 `compare` 共用它，这是 `analyze-demo` 把 `bench_stats.py` 放进 `lib/` 的同一个理由。

实测 8 题 / 8 并发耗时 23 秒，200 题 / 32 并发的实测值记在 §10。

#### 5.4.4 bench（端到端性能回归）

AIPerf 回放 Mooncake trace，命令与容器参数逐字移植自 `profiling-demo/assets/load/aiperf_replay.sh`，包括那四个丢掉就静默出错的设置：uid 映射、两个 dataset mmap 路径、保留镜像自身条目的 `PYTHONPATH`、以及配合 `sitecustomize` 垫片的离线开关（AIPerf 0.12 的 `Tokenizer._resolve_local_snapshot` 不接受本地目录，没有这个 9 行 patch 它会去 HF 拉 tokenizer）。

**每臂两轮**，`bench_rounds` 默认 2。第 1 轮对这份 trace 而言是冷的（radix cache 里没有它的前缀），第 2 轮是热的。`compare` 只做同轮次对比。这是对 `profiling-demo` 那条实测教训的直接回应：同一配置冷热两次差一个量级，所以「一轮就下结论」是错的，而两臂各跑冷热两轮、同类相比，代价只是每臂多两分钟。

### 5.5 `compare`

纯计算，读四个 handoff，产出 `integration_report`。`text.json` 的顶层：

```json
{"generated_at": "...",
 "arms": {"stock": {...}, "patched": {...}},
 "patch": {"operator_id": "...", "files": 1, "apply_mode": "overlay_files"},
 "correctness": {
   "smoke":  {"stock": "pass", "patched": "pass", "verdict": "same"},
   "needle": {"gated": [...], "frontier": [...], "verdict": "same"},
   "evals":  [{"name": "gsm8k", "scored_a": 200, "score_a": 0.95,
               "scored_b": 200, "score_b": 0.945, "delta": -0.005,
               "ci95": [-0.04, 0.03], "verdict": "same"}],
   "prefix_reuse_delta": {"stock": 0.0, "patched": 0.0}},
 "performance": [
   {"round": 1, "metric": "output_token_throughput_tps",
    "stock": 631.2, "patched": 628.9, "rel_delta": -0.0036, "verdict": "same"}],
 "verdict": {"accepted": true, "reasons": []}}
```

判据：

| 线 | 回退的定义 |
|---|---|
| smoke | 任一项 stock 过而 patched 挂 |
| needle（判定档） | 任一深度 stock 过而 patched 挂；或 patched 未达到声明的 token 长度 |
| needle（frontier 档） | 同上，但只影响 `reasons`，不单独否决 |
| llm-eval | Newcombe 区间不含零且 delta 为负 |
| 吞吐 | 相对下降超过 `max_throughput_regression`（默认 0.05） |
| TTFT | 均值或 p95 相对上升超过 `max_ttft_regression`（默认 0.10） |

`report.md` 是同一份内容的人读版本，进 packup 的 `results/`。

### 5.6 `packup`

按 `experiment-result-packup` skill 的 `deliverable_layout.md` 组装：README / REPRODUCE / environment / scripts / results / logs / notes。`collect_env.sh` 复用 skill 里那份，通过 srun 在计算节点上执行。`results/` 放 `report.md`、两臂的 `summary.json` 与 lm-eval 的 `.index`；`logs/` 放压缩后的原始日志。

`REPRODUCE.md` 里「复现成功」的判定收窄为：拉起 stock 部署 + smoke 通过 + 一次 needle 判定档通过 + 一次 200 题 gsm8k 出分。不要求重跑双臂。

---

## 6. 包变量

前缀 `IT_`。前四个无默认值是刻意的：Slurm 作业 ID、它持有哪个节点、该节点的 IP、权重在哪，是一次分配在一个集群上的事实，默认值等于把一台机器的答案当成所有人的，且作业一结束就过期。缺了会在加载期报出文件、行号和变量名。

| 变量 | 默认 | 说明 |
|---|---|---|
| `jobid` / `node` / `node_ip` | 无 | 分配 |
| `model_path` | 无 | 权重目录 |
| `image` | `infera/engine-sglang:glm53-flash` | 引擎镜像。**节点本地**，换节点要重建 |
| `aiperf_image` | `nvcr.io/nvidia/ai-dynamo/aiperf:0.12.0` | |
| `etcd_image` | `quay.io/coreos/etcd:v3.5.14` | |
| `served_name` / `tp` | `glm5.3-flash` / `8` | |
| `router_port` / `worker_port` / `etcd_port` | `8100` / `30000` / `12379` | etcd 不能用 2379，那是节点上 k8s 控制面的 |
| `work_root` | `/data/agent_sys_integration` | 节点本地，不写 `/apps` |
| `patch_target_file` | `srt/models/glm5_next.py` | mock 的目标文件，相对 sglang 包根 |
| `patch_target_symbol` | `Glm5NextDecoderLayer.forward` | mock 的目标函数 |
| `gsm8k_data` | `…/sglang_1p1d_glm5.2/lm_eval/data/test.jsonl` | 实测在位，1319 行 |
| `eval_names` | `gsm8k mixed_prefix_gsm8k` | |
| `eval_examples` | `200` | 全量约 1300 题、每个 eval 约 6 分钟 |
| `eval_threads` / `eval_max_tokens` | `32` / `2048` | |
| `needle_tokens` | `76000` | 较短的一档，约 9.3 个 chunk。不设通过门槛，见 §5.4.2 |
| `needle_frontier_tokens` | `127000` | 较长的一档，同样不设门槛；`0` 关闭 |
| `needle_depths` | `0.02,0.5,0.98` | |
| `aiperf_trace` | `…/sglang_1p1d_glm5.2/aiperf_trace/conversation_trace.jsonl` | 绝对路径而非包内文件，避免每个 attempt zone 复制一份 |
| `trace_end_ms` / `max_conc` / `workers_max` | `120000` / `256` / `16` | |
| `bench_rounds` | `2` | 冷 / 热 |
| `max_error_rate` | `0.05` | `check_bench_report` 阈值 |
| `min_requests` | `100` | 同上 |
| `max_throughput_regression` | `0.05` | `check_no_regression` 阈值 |
| `max_ttft_regression` | `0.10` | 同上 |

首次 bring-up 建议压小：`--var eval_examples=20 --var needle_tokens=31000 --var needle_frontier_tokens=0 --var bench_rounds=1`。

---

## 7. 前置准备（不属于 package）

1. **引擎镜像。** `n04-33` 上已有。换节点必须重建，因为 docker 镜像是节点本地的：

   ```
   docker build -f examples/glm53flash-demo/patches/Dockerfile.sglang.glm53 \
     -t infera/engine-sglang:glm53-flash .
   ```

   实测 9 分 25 秒，构建中有三处断言检查 PR #36507 的 overlay 是否真的落到解释器实际导入的路径上。

2. **AIPerf 镜像**：`docker pull nvcr.io/nvidia/ai-dynamo/aiperf:0.12.0`。

3. **Python 3.12 或更高**。`agent_sys` 声明 `requires-python = ">=3.10"` 但在 3.12 以下无法导入，见 `../temp/bugs/001-…`。仓库自带的 `.venv` 是 3.12。

4. **一个已经持有节点的 Slurm 分配**，登录节点没有 GPU 也无法 ssh 到计算节点：

   ```
   sbatch --parsable --partition=Compute-DCPT --nodelist=<node> --nodes=1 \
     --ntasks-per-node=1 --exclusive --time=09:00:00 --job-name=hold \
     --wrap='srun --ntasks=1 sleep infinity'
   ```

权重不需要准备，理由与 `profiling-demo` 相同：`/apps` 单流冷读实测 921 MB/s，拷到本地盘要把同样的 306 GB 从同一个 NFS 挂载读一遍再写一遍，而节点 3 TB 内存会在第一次加载后把 checkpoint 留在 page cache。

---

## 8. 已知风险与取舍

**挂上了但没被调用的 patch 会给出一个绿色的回归结论。** 这是本阶段最中心的陷阱：一个语法正确、挂载成功、却从未被执行到的 patch，会让两臂的数字完全一致，而 `compare` 会如实报告「无回退」。缓解手段分两层：静态层面 `check_patch_live` 在**运行中的容器里**重算文件 sha256（`__file__` 不是证据，挂载后容器内路径不变）；动态层面要求 manifest 声明 `runtime_marker`，由 `check_patch_live` 在引擎日志里找 `import` 与 `first_call` 两个标记。**局限必须写清楚**：一个不声明 marker 的真实 forge patch 只能拿到静态证据。设计过但本轮不实现的第三层是复用 `profiling-demo` 的 capture + Magpie，在 patched 臂截一个 5 秒窗口，直接看 kernel 符号与占比是否变了；机制现成，代价是每臂多几分钟加一次 graph-off 重启。

**needle 的绝对结果不是这台部署的稳定性质。** 九次测量见 §5.4.2：对长度不单调、对 needle 措辞敏感、对生成预算反向敏感。直接后果是**绝对门槛被取消**，只剩两条事实性底线（长度达标、至少一个深度取回），判定重心整体移到双臂对比。间接后果是：**needle 的通过与否不是模型能力的度量**，它是回归探测器。不要拿本包的 needle 结果去说 GLM-5.3-Flash 的长上下文能力。

**推理解析器让「生成预算」成为一个会伪装成正确性问题的旋钮。** 引擎带 `--reasoning-parser glm45`，答案前面是一大段思考。实测 `smoke.py` 的长生成检查在 512 预算下拿到 `words: 0` / `finish_reason: length`——预算全花在思考上，`content` 是空的；2048 预算下是 153 词答案加 1307 词推理。**答案约占生成 token 的十分之一**，所以任何按答案长度设预算的检查，测到的都是推理解析器而不是它想测的东西。两处检查都把「饿死」与「答错」分开报。

**200 题的 Wilson 区间约 ±5 个百分点。** 一次 quick 跑看不见小幅回退。`eval_examples` 调到全量（约 1300 题）能把区间收到约 ±1.2 个百分点，代价是每个 eval 约 6 分钟。默认取 200 是为了让一轮流水线在半小时量级，这是一个自觉的取舍，写在 `acceptance_*` 的 `watchout` 里。

**temperature 0 不是逐字节可复现的。** 比较答案，不要比较字节。这一条来自 1P1D kit 的 README，本包的 smoke 与 needle 都按答案判定。

**编译型 kernel 不在本轮范围内。** HIP / CK / 汇编的 patch 需要重建镜像，`apply_mode: rebuild` 会快速失败。这不是设计遗漏而是范围声明——重建路径的代价（9 分 25 秒一次，且要把构建产物也纳入证据链）值得单独一轮。

**镜像是节点本地的。** 换节点重跑要先重建镜像。`work_root` 下的 overlay 同理。这一条比听起来更容易踩：Slurm 分配到期后拿到的很可能是另一台机器。

**两次冷启动。** 实测首次从 NFS 加载 819 秒，同会话重启走 page cache 243 秒；加上两次 CUDA graph capture（实测 bs≤128 约 33 秒）。一轮完整双臂流水线预计 45–70 分钟。

**这些节点上跑着 Kubernetes 控制面**，它的 etcd 以 TLS 占住 2379。`etcd_port` 默认 12379，拉起前对五个端口做占用预检。

**`reset_gpus.sh` 原样搬过来会杀掉 `slurmstepd`。** 移植 `profiling-demo` 修正过的版本：只杀名字像遗留推理进程的，显式保护调度器与容器运行时，判定关口改成 VRAM 是否回到基线。

**写入刚结束时 `ls -l` 报的文件大小不可信。** 实测同一脚本内 `du -sb` 与 `ls -l` 相差三个数量级。判定「非空」必须在读取时点用 `stat` / `du` 重新取。

**program task 没有框架级超时**，`agent/backends/program.py` 只有轮询等待子进程；CLI 的 `_settle` 那 1800 秒管的是「整张图是否静默」，不是单任务。每个 entry.sh 自带 timeout。同理，**program task 成功时 stdout 不会被框架保留**（只有非 0 退出码才写尾部 8 KB），所以需要留证的输出都要显式写进 zone 的 `logs/` 或 handoff。

**整张图必须跑进 1800 秒，否则做完的工作会被丢掉。** 这不是设计选择，是 `cli/main.py` 写死的 settle 上限（`../temp/bugs/003`）。默认参数下本包跑不进去：实测 16:05 起跑，两次冷启动加一轮 stock 测量之后，16:35 预算到期时 `serve_patched` 正在收尾——而它其实已经成功（日志末尾是 `MIX_UP_OK`，worker 210 秒起来），一次 306 GB 的部署被拉起、验证健康、然后作废。

`--resume` 接不上：被中断的任务把输出槽留在 open 状态，重派时 store 以 `HandoffStateError: ... is already open by task ...` 拒绝（`../temp/bugs/005`）。resume 能做的是把已封存 handoff 的 validator 补跑一遍——那次它跑了五个，全 PASS——但派发不下去。

所以默认参数只适合有人盯着的单叶子调试，一轮完整两臂跑必须压小到约二十分钟：

```
--var eval_examples=100 --var bench_rounds=1 --var trace_end_ms=60000
```

代价直接落在结论强度上：100 题的 Wilson 区间约 ±7 个百分点，单轮回放没有冷热对照。**一个压测框架的测量上限由它的调度器决定**，这件事本身应该记在 bug 里而不是当作参数调优。

**图中途失败会留下占着显存的容器。** 两个 serve 任务开头都有幂等 teardown 加 VRAM 门禁，所以重跑安全；不重跑就要手工 `docker rm -f glm53_int glm53_int_etcd`。框架没有 per-task 清理钩子。

**handoff 的发布封印会拒绝内容文件里的绝对路径。** 沿用 `profiling-demo` 的三条对策：站点路径用 `@NAME@` 占位符（`${NAME}` 不行，`}` 不在 `locality._CANDIDATE` 的 lookbehind 排除集里，剩下的 `/assets/...` 又成新候选，而 `@` 在排除集里）；原始日志压缩后入包（封印跳过无法按 UTF-8 解码的文件）；`reproducible` 带 `command` 而不是 `script`。另外 `agent/gate.py` 要求 `script`/`command`/`entry` 带可执行位。

**这条规则会咬到散文，不只是路径。** 封印的判据是「两段以上的斜杠序列，且前一个字符不在排除集里」，而 `>` 不在排除集里——所以 `REPRODUCE.md` 里写 `<this package>/assets/serve` 这种带尖括号占位的说明，会被读成绝对路径 `/assets/serve`。`analyze-demo` 记过同一个陷阱（那边是 `<operator_id>/scripts/x.py`）。实测后果：`packup` 把整个交付物 45 个文件写完，然后在最后一行被拒绝。改法是让占位符落在 shell 变量上（`$PKG/assets/serve`，`G` 在排除集里），并且连解释这件事的那句话本身也要避开该形状。写完任何面向人的文档之后，用 `assets/lib/redact.py <dir> NOOP=/nonexistent` 重扫一遍是值得的——它跑的就是封印自己的规则。

---

## 9. 目录布局与运行方式

```
integration-demo/
├── DESIGN.md                          本文档
├── README.md                          跑法、图的形状、预期产物
├── main.yaml                          根 closure + subgraph
├── shared.yaml                        共享的 program agent，包变量的唯一声明处
├── steps/
│   ├── patch.yaml                     kernel_patch / patch_overlay kind
│   │                                  + check_patch_shape / check_overlay_applies
│   │                                  + seed_patch / apply_patch task
│   ├── stock.yaml                     deployment_stock / acceptance_stock / bench_stock
│   │                                  + check_service_live / check_acceptance / check_bench_report
│   │                                  + serve_stock / measure_stock task
│   ├── patched.yaml                   deployment_patched / acceptance_patched / bench_patched
│   │                                  + check_patch_live
│   │                                  + serve_patched / measure_patched task
│   └── verdict.yaml                   integration_report / integration_packup
│                                      + check_no_regression / check_packup_shape
│                                      + compare / packup task
└── assets/
    ├── lib/
    │   ├── remote.sh                  srun 远程执行封装
    │   ├── redact.py                  站点路径 -> @NAME@，再跑一遍封印自己的规则
    │   ├── store.py                   validator 读已发布 handoff
    │   ├── eval_stats.py              Wilson 区间 + Newcombe 差值区间
    │   └── patchkit.py                manifest / mounts 的读写与 sha256 校验
    ├── main.task/readme.md
    ├── seed_patch.task/               readme.md + entry.sh + seed.py
    ├── apply_patch.task/              readme.md + entry.sh + apply.py
    ├── serve_stock.task/              readme.md + entry.sh
    ├── serve_patched.task/            readme.md + entry.sh
    ├── measure_stock.task/            readme.md + entry.sh
    ├── measure_patched.task/          readme.md + entry.sh
    ├── compare.task/                  readme.md + entry.sh + compare.py
    ├── packup.task/                   readme.md + entry.sh + packup.py
    ├── serve/                         两个 serve task 共用
    │                                  round.sh + mix_up.sh + mix_worker.sh
    │                                  + mix_smoke.sh + reset_gpus.sh
    ├── accept/                        两个 measure task 共用
    │                                  measure.sh + smoke.py + needle.py
    │                                  + probe.py + lm_eval.sh
    ├── bench/                         aiperf_replay.sh + summarise.py
    │                                  + pythonpath/sitecustomize.py
    └── <八个>.validator/              readme.md + entry.sh + check.py
```

`assets/serve/`、`assets/accept/`、`assets/bench/`、`assets/lib/` 不带 `.task` 后缀，因为它们是共用脚本而不是 body 目录，框架只按 `<name>.task` / `<name>.validator` 找 body。另外**共用脚本不能和某个 closure 同名**：`spec_loader` 按文件名匹配 closure 名来解析 body，同名会让加载期以「有两个候选」拒绝。

### 运行方式

```
export PATH="<repo>/.venv/bin:$PATH"
AGENT_SYS_NO_PERMISSIONS=1 agent-sys run \
  --package agent_sys/examples/llm_e2e_performance_optimization/integration-demo \
  --var jobid=<job id> \
  --var node=smci355-ccs-aus-n04-33 \
  --var node_ip=10.235.192.139 \
  --var model_path=/apps/qiongzhu/models/GLM-5.3-Flash-FP8
```

`AGENT_SYS_NO_PERMISSIONS=1` 是 mission 要求的。`permissions.grants` 照写不误——它们是「这个任务碰了什么」的答案，现在写下来意味着以后打开校验是一次配置变更而不是一次考古。

### 与上下游衔接

**上游。** stage 4（kernel optimization）落地后：删掉 `seed_patch` 叶子与 `steps/patch.yaml` 里它那一段，把 `apply_patch` 的 `froms` 指向 forge 阶段的产出任务，`kernel_patch` 这个 kind 的定义两包保持一致即可合图。kind 定义放在 `steps/patch.yaml` 而不是 `shared.yaml`——合图时它整体搬到上游包去，留在这边的只有消费方。

**下游。** 没有下游，本阶段的 `integration_report` 就是整条链路的验收结论。`integration_packup` 是交给人的交付物。

---

## 10. 落地顺序与实测记录

分五步，每步都有可验收的中间态：

1. ~~**手工打通关键环节**（不涉及 agent_sys）~~ —— **已完成**。
2. ~~**只做 `seed_patch` + `apply_patch` 两个叶子**~~ —— **已完成**。这一步暴露了容器路径必须用占位符（§3.2）。
3. ~~**加 `serve_stock` 与 `measure_stock`**~~ —— **已完成**。
4. ~~**加 `serve_patched` 与 `measure_patched`**~~ —— **已完成**。
5. ~~**加 `compare`、`packup` 与两个 validator**~~ —— **已完成**。

### 全图实测（`smci355-ccs-aus-n04-33`，2026-08-31，run `20260831T185620-cd3305`）

八个任务全部 succeeded，十种 handoff 全部产出，八个 validator 全部 PASS：

| 任务 | handoff | validator |
|---|---|---|
| `seed_patch` | `kernel_patch` valid | `check_patch_shape` PASS |
| `apply_patch` | `patch_overlay` valid | `check_overlay_applies` PASS |
| `serve_stock` | `deployment_stock` valid | `check_service_live` PASS |
| `measure_stock` | `acceptance_stock` valid | `check_acceptance` PASS |
| | `bench_stock` valid | `check_bench_report` PASS |
| `serve_patched` | `deployment_patched` valid | `check_patch_live` PASS、`check_service_live` PASS |
| `measure_patched` | `acceptance_patched` valid | `check_acceptance` PASS |
| | `bench_patched` valid | `check_bench_report` PASS |
| `compare` | `integration_report` valid | `check_no_regression` PASS |
| `packup` | `integration_packup` | `check_packup_shape` PASS（内容已写完，封存被 settle 上限截断，validator 单独跑过） |

**判定为 ACCEPTED，而这正是正确答案**：被测的是语义等价的 mock，只加两行日志、不改任何算术，所以「无差异」是已知的期望值。一个在这种输入下报出回退的对比，问题在判据而不在被测对象——这就是 mock 存在的意义。

```
smoke      arithmetic / long_generation / workers / models / engine_log   全部 same
needle     31k token 三个深度，两臂均取回                                  全部 same
gsm8k                0.95 -> 0.90    差值 95% 区间 [-0.26, +0.15]           same
mixed_prefix_gsm8k   1.00 -> 1.00                 [-0.16, +0.16]           same
吞吐               762.28 -> 770.67 tok/s   +1.1%                          same
TTFT 均值         3389.31 -> 3465.58 ms    +2.3%                          same
token 间延迟       151.59 -> 143.40 ms     -5.4%                          same
回放请求数            166  ->    166                                    context
```

eval 的区间这么宽，是因为这一轮用 20 题跑以装进 settle 预算；区间就是 20 题能说的全部。

**patch 被证明真的执行了，不只是挂上了。** `check_patch_live` 读到：容器内文件哈希等于 `sha256_patched`；引擎日志里 import marker **9 次**（8 个 TP rank 加驱动进程）、first_call marker **8 次**——真实请求进入了被 patch 的代码路径。这是本阶段整套设计要拿到的那份证据。

各步骤实测耗时（CUDA graph 开）：`serve_stock` 257 s（checkpoint 在 page cache），smoke 22 s，needle 两档 52 s，probe 12 s，`gsm8k` + `mixed_prefix_gsm8k` 各 100 题共 62 s，一次 60 秒窗口的 trace 回放 85 s、166 个请求。一轮完整两臂在上述缩减参数下约 18 分钟。

### 五轮全图，四轮各暴露一处设计错误

每一条都已修复，也都写进了本文对应章节，而不是悄悄改掉：

1. **`froms` 不参与调度**（§4.1，`../temp/bugs/004`）。两个 serve 任务被同时派发，撞在同一个容器名上。
2. **probe 作为执行门禁会让一次波动毁掉整臂**（§5.4.3）。
3. **推理解析器让生成预算伪装成正确性问题**（§5.4.1、§8）。512 与 2048 两次都拿到空答案加一千多词推理。
4. **patched 臂自己的证据含容器路径**（§8）。`serve_patched` 把整个 handoff 写完，然后在最后一行被封印拒绝。这一条最难归因：框架把它报成「running」直到 settle 上限，中间空了十七分钟，两次完全一致——单独跑一遍 body 才看到 `ABORT: evidence still names paths the handoff seal will refuse`。

第五条不是本设计的问题而是框架的：跑不进 1800 秒的图也无法用 `--resume` 接上，被中断的任务把输出槽留在 open 状态（`../temp/bugs/005`）。

---

## 11. 需要确认的问题

1. **`kernel_patch` 的契约是否可以就这样提给 stage 4？** §3.2 是本文单方面定义的。如果 KernelForge 的 `forge_result.json` 已有既定 schema，`manifest.json` 的 `expect` 块应当直接引用它而不是另起一套。
2. **`runtime_marker` 该不该是必填？** 现在是可选，代价是不声明 marker 的 patch 只有静态证据（§8 第 1 条）。要求 stage 4 在优化后的 kernel 里留一个一次性标记，成本很低而收益是把「patch 真的被执行了」变成硬证据。
3. **两档 needle 是否都值得保留。** 现在两档都不设门槛、都只进双臂对比，每臂多花约 30 秒。留两档的理由是「差异只在某一档出现」本身是信息；如果第一轮实测发现两档结论总是一致，就该只留一档。

3b. **needle 的生成预算该固定在哪。** 256 会饿死答案，2048 会让模型把答案说没，两者都实测过。现在取 2048 并把「饿死」单独报出来，但这不是一个有依据的最优值——它只是「至少不会把配置问题伪装成正确性问题」。要定下来需要在若干预算上各扫一遍三个深度。
4. **是否要加第三层 patch-live 证据**（capture + Magpie 看 kernel 符号）。机制现成，代价是每臂多一次 graph-off 重启。
5. **`max_throughput_regression` 默认 0.05 是否合适。** 这个数字目前没有实测依据——需要先测出同一臂两次运行的自然离散度，才知道 5% 是宽是严。落地后第一轮就能拿到这个数。
