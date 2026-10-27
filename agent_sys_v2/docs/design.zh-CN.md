# agent_sys v2 设计：产品、前端与后端

本文写 agent_sys v2 的产品定位、架构和规则。安装、运行和开发见 [README](../README.zh-CN.md)。文中 `<p>` 指项目名，路径除特别说明外都相对于 `AGENT_SYS_HOME`。

## 0. 一页概览

agent_sys v2 是 Infera 的**性能工程 agent 工作台**。一名工程师在网页上指挥一组 Claude Code agent，在 GPU 集群上找出并消除 Infera 部署的瓶颈。瓶颈可以在任何一层：拓扑与并行、引擎参数、CPU 路径、通信、KV 传输、kernel。每个结论都要有实测证据，每个 agent 都看得见、管得住。

```
 浏览器 ──ssh -L──► agent-sys-v2 serve（FastAPI，Unix socket，tmux 窗口）
                     │  扫描文件 → 内存模型 → REST + SSE
                     │  调度：依赖、并发名额、机器占用
                     │  跟踪：每 0.5 s 读运行中任务的 events.jsonl
                     ▼
              每次运行一个 runner 进程（tmux 窗口）
                     │  claude-agent-sdk → claude CLI（每个任务一个会话，多轮 resume）
                     │  与 server 只通过 run 目录里的文件通信
                     │  agent 报告完成后：执行验收检查，失败就把输出发回同一会话
                     ▼
     projects/<p>/tasks/*.md  ·  workspaces/<任务>/{progress.md, report.md, results/*.json, .agent_sys/}
```

和 v1 的 agent_sys（任务包运行时）相比：

| | v1（任务包） | v2 |
|---|---|---|
| 工作流 | 事先声明好的静态 DAG（YAML，5 类对象） | 一个任务就是一份 prompt。图由人和 agent 在过程中长出来：agent 起草子任务，人批准 |
| agent 的角色 | 照着 brief 执行步骤（"do not invent a method"） | 自己决定查什么、试什么；skills 提供可靠的工具和方法 |
| 质量保证 | handoff schema + validator，管的是格式 | 实测结果台账 + A/B 判定规则 + 验收检查，管的是结论 |
| 可观测 | 状态变化时打一行；agent 消息丢弃 | 每个 agent 的实时输出、花费、当前命令、后台等待、机器占用、实验曲线 |
| 可控 | 只能 Ctrl-C 结束整个 run | 对单个任务发消息、停止、终端接管、追加预算、审核、回答问题 |

## 1. 定位与原则

**定位**：一名工程师加一群 agent 的工作台，人只做决策和验收。它不是给人用的项目管理工具，也不是自动驾驶的优化器。

**原则**：

1. **用户体验优先。** 需要人处理的事集中在一个页面，一行一件，带好主按钮。新建任务就是写一段话，不用学一套 YAML 本体论。界面和文件始终一致；不开界面，也能照同样的约定工作。
2. **信任 agent，核验证据。** 不限定 agent 的步骤；限定的是证据的标准，包括结果格式、A/B 判定规则和验收检查，并且由 harness 而不是 agent 自己去执行检查。两条经验：昂贵的 gate 前先放便宜的检查；有意思的失败不是崩溃。
3. **可观测。** 运行的每一帧都落盘（`events.jsonl`），界面实时展示。注入了什么、花了多少、在等什么，都写在 `launch.md` 和卡片上。
4. **可控。** 任何一个 agent 都能单独发消息、停止、接管、加预算、审核。机器占用由调度器管，不靠口头约定。
5. **简单。** 文件就是状态，server 可以随时重启。一个进程模型，一套状态词汇。新功能先问：能不能让产出更可信？能不能省人的时间？

## 2. 用户与核心循环

**用户**：
- **性能工程师**（主要用户）：给定模型 × 硬件 × 引擎 × 拓扑 × 负载，把 goodput 提上去。节点稀缺，希望几个实验并行；不想守着屏幕，早上来看结果，而不是看到一台闲置了一夜的机器。
- **评审者**：看结论和证据，决定哪些改动进 overlay 补丁或 PR。
- **团队**：skills、经验库和基线在项目之间共享。

**核心循环**（模板 `infera-perf`，按轮推进；第 1 轮的 5 个任务由模板在建项目时预置）：

```
 ① intake/rN       预填问卷 → 你回答 → 写项目的规则、目标、机器、工作点 → brief（审核）
        │
 ② survey/rN ∥ baseline/r1   对标曲线进台账（reference）、配置差异表 ∥ v0 曲线、噪声、快速验证标定
        │
 ③ profile/rN      从便宜的层做起：时间账 → 资源利用 → 只对瓶颈层做 kernel / CPU → candidates.yaml（审核）
        │
 ④ plan/rN         候选排序、GPU 预算和机器分配（审核）→ 起草 experiment 子任务和 close/rN
        │
 ⑤ experiment × N  一个大杠杆或一组同层小杠杆：快速验证迭代 → 工作点单点 A/B → 判定；按机器并行
        │
 ⑥ close/rN        depends_on {plan/rN, kind: children}：子任务全部完成或取消后开始；集成 → 整条曲线 →
                   封版 vN（不劣于 v(N−1) 时）→ 复盘 → 起草 intake/r(N+1)（带建议的增量问卷）
```

和固定流水线的区别：③决定④⑤做什么，⑤⑥的结果又决定下一轮。kernel 优化只是③的一种，跟 tokenizer workers、调度策略、P:D 比、Mooncake 参数一视同仁，由②的证据决定先做哪个。

## 3. 概念模型

一共 7 个概念，每个都落在文件上：

| 概念 | 是什么 | 文件 |
|---|---|---|
| **项目**（project） | 一次优化活动：目标、机器、台账配置、规则、经验 | `projects/<p>/project.yaml`、`rules/`、`knowledge/` |
| **任务**（task） | 一份 prompt 加一个工作区；任务类型就是看板上的泳道 | `tasks/<类型>/<名字>.md`、`workspaces/<类型>/<名字>/` |
| **运行**（run） | 任务会话的一轮：启动参数、注入内容、全部事件、结束记录 | `workspaces/…/.agent_sys/runs/<run>/` |
| **检查**（check） | agent 报告完成后由 harness 执行的验收命令 | 任务 frontmatter 的 `checks`；结果在 `runs/<run>/checks.json` |
| **结果**（result） | 一个测量点，统一 schema | `workspaces/…/results/*.json`（`schema: infera.result/v1`） |
| **机器**（machine） | 项目可用的节点；任务声明要用哪些，调度器保证互斥 | `project.yaml` 的 `machines`；任务 frontmatter 的 `machines` |
| **skill** | 可复用的操作手册和脚本，例如部署、测量、分诊、A/B | `src/agent_sys_v2/skills/<名字>/SKILL.md`；项目自己的放 `projects/<p>/skills/` |

另外还有两个约定：**规则**（全局和项目的 `rules/*.md`，每次运行都注入）和**经验**（任务把候选写在工作区的 `lessons/`，人采纳后进经验库，经验库的 `INDEX.md` 每次运行都注入）。

**依赖**：任务之间的边有 5 种。`input`（产出依赖）、`code`（代码基线）和 `children`（等上游和它的全部子任务，见 §6.1）会阻塞下游，参与环检测；`review`（评审覆盖）和 `subtask`（父子关系）不阻塞。边的来源按优先级从低到高：prompt 里写到的其他任务路径（`tasks/<类型>/<名字>`、`workspaces/<类型>/<名字>` 或 `@<类型>/<名字>`，记为 `input`；如果是本任务的 parent，记为 `subtask`）、frontmatter 的 `parent`（`subtask`）、frontmatter 的 `depends_on`（写 `类型/名字`，或 `{task, kind, label}`，kind 缺省为 `input`）。通过界面新建、修改或改名任务时，会形成阻塞类环的写入被拒绝；直接改文件形成的环，环上每个任务都有记录提醒。

### 3.1 任务状态

每个状态都用形状、文字、颜色三样同时表示。

| 状态 | 含义 |
|---|---|
| 草稿 draft | 有 prompt，未托管（agent 起草的子任务也在这里） |
| 待开始 queued | 卡片写明在等什么：项目已归档、prompt 有错误、上游未完成、等上游的子任务、等你审核模板、自动开始已关闭、**等机器**（写明被谁占着）、名额已满（§5.5） |
| 运行中 running | 执行中 / 等待中（后台命令在等作业，不花 token）/ 终端接管 |
| 待确认 needs_you | agent 提问、`questions.md` 有问题待答、被停止、运行中断（含后台等待被中断）、预算用完、纳入托管时还在进行中，或**验收未通过** |
| 待审核 in_review | 完成了，且设置了 `review: true` |
| 已完成 done | 新完成的任务带"未查看"标记 |
| 已取消 cancelled | |
| 其他 unknown | 未托管任务的 progress.md 状态行不认识，或 state.json 无法解析 |

"已完成"由两件事共同决定：agent 在 `progress.md` 里写了已完成，并且 `checks` 全部通过。没有配置检查的任务只看第一条。已完成的任务回答追问（再跑一轮、报告没改）时，检查不重跑，任务仍是已完成。

托管任务的状态在 `.agent_sys/state.json` 里，只由系统写；未托管任务的状态按 `progress.md` 开头的"状态:"行推导。等待原因、运行中的子状态这类能从别的文件推出来的信息不存进 state.json，扫描时现算。

### 3.2 机器占用规则

- 托管任务在运行中和待确认时占用它声明的机器，直到进入待审核、已完成或已取消；已完成的任务再跑一轮（回答追问）时重新占用。
- 待确认也算占用：等人回复期间，部署往往还在节点上。
- 机器按主机名识别，规则跨项目生效。
- 一个待开始的任务，只要它要的机器有一台被别的任务占用，调度器就不会启动它；卡片写明"等机器：node-a（被 experiment/e3 占用）"。任何方式开始一轮运行（开始、发消息、继续、接管、追加预算）都会检查机器，被别的任务占用时拒绝并说明占用者；任务自己已经占着这些机器（例如在待确认时收到回复）也照样检查。要越过这条规则，就改任务的机器列表。
- 正常情况下一台机器只有一个占用者。规则被绕过时（例如直接改文件，给一个待确认的任务加上别人占着的机器），所有占用者都会列出来，看板和首页一个占用者一项。
- 占着机器又在待确认里等了 30 分钟以上的任务，首页和看板会标出"闲置占用"和时长。这是对"agent 睡着、节点空转一夜"的直接回应。

## 4. 前端设计

技术：Preact + htm，不需要构建，ES module 直接加载。和 API 同源；脚本只从本站加载（CSP 见 §5.9）。只有一条 SSE 推送变化，运行输出按通知去取增量。中英双语（`agent_sys.yaml` 的 `language`，默认英文；界面右上角可切换），默认暗色，可切换为亮色或跟随系统。

### 4.1 页面

| 页面 | 路由 | 内容 |
|---|---|---|
| **首页** | `#/` | 顶部三张卡片：项目列表（一行一个，带待确认、待审核、**待审核模板**等标记；"新建项目"收在卡片标题旁的按钮里）、新建任务（选项目后在首页打开新建任务抽屉，快捷键 n）、运行中（附各项目今日花费）。下面是**机器**（所有项目被占用的机器和闲置告警），然后是待确认卡片（一行一件，按原因给主按钮：回复 / 回答 / 查看检查 / 追加预算 / 继续）、待审核、最近完成，空的队列折叠成一行 |
| **看板** | `#/p/<p>` | 头部：标题、**目标**、**基线和当前最佳（Δ%）**、自动开始、名额、新建任务。模板预置的任务还没开始时，头部下面是**待审核模板**：按执行顺序列出这些任务、项目规则和设置的入口，以及"开始运行"。**机器条**：每台机器显示被哪个任务占着（每个占用者一项）、占了多久、占用者的状态。待处理摘要、搜索和筛选。泳道 × 日期的依赖图（按执行顺序从左上到右下：上游总在下游之上，同时创建的任务按依赖深度和泳道排；依赖线是只走泳道之间和行之间空隙的直角折线，同一空隙里的线各占一条轨道并按少交叉排序，可由其他依赖推出的产出、代码基线依赖默认不画，"视图"里或图上方可切换为全部显示），卡片上有状态、机器、检查 ✓/✗、结果 Δ%。下方是任务列表 |
| **台账** | `#/p/<p>/ledger` | 头部：主指标及方向、基线值（`ledger.baseline` 指定的任务或文件没有可用结果时给出提醒）、最佳值和 Δ%、结果数（作废、格式错误）。项目配置了 X 指标时，先是**曲线**：每个系列一条 Y–X 折线（沿前沿），标出工作点和各系列在工作点的值与 Δ，附系列表。然后是趋势图：每个结果一个点，形状和颜色表示判定（接受 / 拒绝 / 噪声 / 需重跑 / 未判定 / 作废，基线是菱形）；不按工作点比较时还画基线虚线和"目前最佳"阶梯线。最后是结果表（台账 Δ 和判定 Δ 分两列；行上标"判定未核实"及原因、"相对 v0 判定"、"复测"）和格式错误的文件。点一行或一个点，打开任务抽屉并定位到这个结果文件 |
| **任务抽屉** | `?t=<p>/<类型>/<名字>` | 头部：状态和原因、标题、花费/预算、模型、进度（已勾选/全部）、**机器**、**检查摘要**。三个 tab：**运行输出**（实时，检查的过程和结果也在流里）；**工作区文件**（报告审核、questions.md 表单、交接清单、验收检查的每次尝试、结果文件）；**任务配置**（prompt、上游任务、类型和名字、机器、检查、agent 配置）。底部一个输入框：运行中发的消息在当前命令后读到，其他状态发消息就接着跑。另有停止、在终端中接管、追加预算 |
| **新建任务** | `?new=<p>` | 选类型（prompt 按类型骨架预填，检查也一起预填），选上游任务（边写边预览 prompt 和所选上游给出的依赖），选机器，设审核、预算、模型和 effort；可以直接排队，也可以存为草稿 |
| **规则 / 经验 / 设置** | `#/rules`、`#/knowledge`、`#/settings`；项目的在 `#/p/<p>/…` 下 | 规则页显示每次注入的各段及大小，包括 **skills 索引**；经验页处理候选的采纳、改写为规则和丢弃，以及编辑经验库；设置页（全局：界面语言、给 agent 的语言、全局名额；项目：标题、目标、机器、台账、自动开始、名额、任务默认值） |
| **已归档项目** | `#/archived` | 归档的项目不显示在首页，任务不会自动开始，也不能新建任务或启动运行 |

### 4.2 交互要点

- **agent 起草的子任务**：显示为草稿卡片，挂在父任务下。父任务抽屉里有"子任务草稿 N 个，全部纳入托管"，一次批准一组；也可以逐个修改后再批准。
- **回答问题**：agent 有多个问题时按约定写 questions.md 表格（`# | 问题 | 建议 | 你的意见`，单元格里的 `|` 写成 `\|`）。卡片主按钮"回答 N 个问题"打开表单，你的回答写回表格的"你的意见"格（没填的行按建议，其余内容原样保留），并告诉同一会话接着做；任务此时不能开跑（prompt 有错误、预算用完、机器被占用）就拒绝提交，什么也不写；本轮写的 questions.md 里没有"你的意见"表时，主按钮是"回答问题"，打开这个文件，直接回复即可。
- **验收未通过**：卡片主按钮是"查看检查"，打开运行输出并定位到检查结果。回复"继续"后，agent 在同一会话里接着修。
- **机器冲突**：卡片的等待原因直接链接到占用者。占用者在待确认时，首页的那一行带"闲置占用"标签。
- **台账和任务**：卡片上的结果徽章显示最好的 Δ；台账的行和点打开任务抽屉，定位到对应的结果文件。
- **继续全部中断任务**：机器重启之类的原因让多个任务同时运行中断时，首页一个按钮让它们全部接着跑。
- 不做：燃尽图、故事点、多用户权限。

### 4.3 视觉

- 页面底色带一点冷绿；
- 状态色：待确认橙、运行中蓝、待审核紫、完成绿；
- logo：圆角方块里一条带节点的折线逐级上升（一串任务把指标推高），也是 favicon；
- 卡片左侧色条标状态；唯一会动的是运行中图标（一段弧绕着中心点转）；首屏加载时 logo 的折线自己画出来，页面内加载是一条扫过的细光带；遵守 `prefers-reduced-motion`。

台账的判定：接受用完成绿，拒绝用警告红的叉，噪声用灰，需重跑用待确认橙，作废是空心点，基线是菱形；曲线每个系列一个分类色。

## 5. 后端设计

### 5.1 进程

| 进程 | 职责 | 生命周期 |
|---|---|---|
| `agent-sys-v2 serve` | 扫描、调度、跟踪运行、REST + SSE、提供前端 | 在 `tmux -L $AGENT_SYS_TMUX`（默认 `agent_sys`）的 `server` 窗口里。可随时重启，正在跑的任务不受影响。一个 `AGENT_SYS_HOME` 只能有一个 server（home 里的锁文件） |
| runner（`python -m agent_sys_v2.runner <run_dir>`） | 一轮会话：启动或恢复 claude 会话，转发消息和停止指令，记录事件；一轮结束时如果 agent 报告完成，执行检查 | 每轮一个，各自在一个 tmux 窗口里；会话空闲、没有后台命令、收件箱已读完（检查也跑完）时退出 |
| 终端接管（`runner <run_dir> --terminal`） | 包装进程在 tmux 窗口里对同一会话开交互式 claude；claude 退出后，progress.md 已完成就执行一次检查，再写结束记录 | 用户退出 claude 或在界面上点"结束接管"后结束；server 按 progress.md 和检查结果接回 |

server 和 runner 只通过 run 目录里的文件交换信息：

| 文件 | 写入方 | 内容 |
|---|---|---|
| `launch.json` | server | 启动参数（模型、预算、环境变量、要执行的检查、本轮开始前任务的状态） |
| `system.md` | server | 注入内容 |
| `launch.md` | server | 给人看的注入清单、首条消息和等价命令 |
| `inbox.jsonl` | server 追加 | 用户消息、停止指令 |
| `events.jsonl` | runner | CLI 的原始帧，加上 runner 自己的事件 |
| `run.json` | runner | 开始记录和结束记录（退出原因、会话累计花费） |
| `checks.json` | runner | 每一次检查尝试的结果 |
| `stderr.log` | runner | runner 自己的错误输出 |

### 5.2 文件布局

```
$AGENT_SYS_HOME/                       默认 ~/agent_sys_home
  agent_sys.yaml                       全局配置（language、protocol_language、max_running、claude_config、cli_path）
  claude/                              agent 自己的 Claude Code 配置目录（CLAUDE_CONFIG_DIR，§5.9）：settings.json、
                                       .claude.json 每次启动生成或补齐，projects/ 链接到 ~/.claude/projects
  rules/*.md                           全局规则
  knowledge/{INDEX.md,entries/}        全局经验库
  projects/<p>/
    project.yaml                       目标、机器、台账、skills、agent 默认值、自动开始、并发、泳道顺序
    rules/*.md  knowledge/  skills/    项目的规则、经验、自带 skills
    task_types/<类型>.md                prompt 骨架（frontmatter 可带默认的 review、checks）
    tasks/<类型>/<任务>.md               任务 prompt（frontmatter：title, summary, depends_on, parent, review, agent,
                                       machines, checks, check_retries）
    workspaces/<类型>/<任务>/
      progress.md report.md issue.md questions.md lessons/ log/ results/
      .agent_sys/state.json            托管状态（只有系统写）
      .agent_sys/runs/<run>/           一轮的完整记录
    repos/                             代码仓库，不扫描
    .trash/                            删除的任务（prompt 和工作区一起移进来）
```

`project.yaml` 示例（`infera-perf` 模板填好之后）：

```yaml
title: GLM-5.2 4P4D AgentX
template: infera-perf
goal: "AgentX 工作点 P90 interactivity ≥ 30 下的输入 tok/s/GPU +10%"
machines:
  - {name: smc-a16-19, gpus: 8, note: MI355X}
  - smc-a16-20
round: 1
version: v0
ledger:
  primary: input_tput_per_gpu
  higher_is_better: true
  x: p90_intvty
  baseline_series: v0
  working_point: {x: 30}
  noise_pct: 2            # A/B 的噪声带和接受线（%，默认 2、3）：只由用户改，命令行只能提高接受线，运行开始时快照（§5.6）
  accept_pct: 3
skills: [infera-deploy, infera-bench, infera-quality-gate, infera-profile, infera-diagnose, infera-experiment, infera-hygiene]
autorun: true
max_running: 3
lanes: [intake, survey, baseline, profile, plan, experiment, close, develop, review]
agent: {model: opus, effort: high, budget_usd: 200, extra_args: []}
```

模板预置的任务还没审核时，`project.yaml` 里还有一行 `template_review: true`（§6.1）。

### 5.3 模块

一个 Python 包 `agent_sys_v2`（目录 `Infera/agent_sys_v2/`，src 布局，与 v1 的 `agent_sys/` 并列）。

| 模块 | 职责 |
|---|---|
| `model.py` | 数据类型（Project、Task、Edge、TaskType）、frontmatter 读写、state.json 和 run.json 的格式、未托管任务的状态推导、文件读写工具 |
| `scan.py` | 扫描一个项目：任务、依赖边、等待原因、记录提醒；Watcher 按 stat 发现变化 |
| `ops.py` | 写操作：新建项目（按模板）、类型、任务，改名，删除，审核，纳入托管和批量纳入子任务草稿，归档，开始运行，泳道顺序，设置 |
| `sched.py` | 调度：启动待开始的任务（依赖、名额、机器） |
| `machines.py` | 解析机器列表；占用表；闲置占用；互斥检查 |
| `runs.py` | 开始、消息、停止、接管、预算、取消、一轮结束后的状态迁移；Monitor（server 跟踪运行中任务的循环） |
| `runner.py` | 一轮会话的进程；检查的执行和反馈；终端接管的包装进程 |
| `events.py` | 读 events.jsonl：给运行输出的精简事件（分页、单条全文），以及跟踪运行中会话的状态 |
| `procs.py` | 在 /proc 里找一轮运行的进程（runner、claude 会话、检查）并结束它们 |
| `replay.py` | 回放录好的 events.jsonl，不调用模型（测试、演示、界面开发） |
| `launch.py` | system.md（约定、规则、skills 索引、经验索引）、首条消息和后续消息、SDK 参数、launch.md |
| `claude_config.py` | agent 的 Claude Code 配置：专用目录、env 和 settings 键的白名单、`projects/` 软链、首次启动状态、退路 `claude_config: inherit`（§5.9） |
| `tmux.py` | agent_sys 的 tmux server 和窗口；交给子进程的干净环境 |
| `checks.py` | 解析 checks、执行（超时、输出截断）、记录和汇总、失败反馈 |
| `ledger.py` | 读结果文件、校验 schema、算 Δ（参照函数 `reference`）、最佳值和曲线；命令行 `ledger check`、`ledger verify`（可当检查用）和 `ledger agentx`（AgentX 输出转结果文件，记下来源） |
| `verdict.py` | A/B 判定：判定函数（ab_judge 和复算共用）、`verdict.inputs`、两类作废、口径和快照、从 AgentX 原始输出复算（`ledger verify`，台账的"判定未核实"） |
| `perfcheck.py` | `infera-perf` 模板一轮里各任务的字段检查，供验收检查调用 |
| `skills.py` | 选出项目可用的 skills，生成 skills 索引 |
| `questions.py` | questions.md 表单：读表格、写回答、判断是不是本轮的问题 |
| `knowledge.py` | 规则页和经验页：规则读写、经验候选的采纳和丢弃、经验库和 INDEX.md |
| `files.py` | 工作区文件的列目录、读、写 |
| `i18n.py` | 中文源文字的英文翻译（界面和给 agent 的文字） |
| `server.py` | REST + SSE，提供前端；Host/Origin 校验 |
| `cli.py` | `agent-sys-v2 serve / new / --version` |

另有 `web/`（前端，见 `web/README.md`）、`templates/`（注入的约定 `_common/rules/protocol*.md`，项目模板 `blank`、`infera-perf`）、`skills/`（内置 skills）。

### 5.4 一轮运行和验收检查

```
start_run → 写 run 目录 → tmux 开 runner
runner：connect / resume → 发首条消息或你的消息 → 转发收件箱 → 记录帧
  一轮结束（会话空闲、没有后台命令、收件箱已读完）：
    progress.md 没写"已完成"            → 退出（server 判为待确认：提问 / 有问题待答）
    写了"已完成"，只是回答了追问        → 退出，不跑检查（仍为已完成）
    写了"已完成"，且有检查               → 逐条执行检查（runner 事件 checks / check，带 phase，流进运行输出）
        全部通过                         → 退出（server 判为已完成或待审核）
        有失败，且还有重试次数            → 把失败的检查和输出尾部作为一条消息发回同一会话，等它再次标记完成
        有失败，且重试次数用完            → 退出（server 判为待确认 · 验收未通过）
```

- **执行**：每条检查用 `bash -c` 执行，cwd 是项目目录，按顺序全部执行（一条失败不跳过后面的）。环境变量有 `AGENT_SYS_PYTHON`（运行 agent_sys 的 Python，用它调用 `agent_sys_v2.ledger` 等工具，不依赖 PATH）、`AGENT_SYS_PROJECT`、`AGENT_SYS_TASK`、`AGENT_SYS_WORKSPACE`、`AGENT_SYS_MACHINES`（逗号分隔）、`AGENT_SYS_SKILLS`（内置 skills 的目录）；agent 自己的环境里也有这几个变量。默认超时 600 s，超时就杀掉整个进程组；输出只留最后 4000 字符。
- **事件和记录**：runner 把 `checks`（phase start / end）和每一条 `check`（phase start / end）写进 events.jsonl，运行输出实时显示；每一次尝试追加到 `checks.json`。server 读最后一次尝试，决定已完成还是验收未通过；任务抽屉和卡片都读它。
- **反馈**：检查列在给 agent 的首条消息里。失败时发回 agent 的消息写明哪几条检查失败、退出码或超时、输出尾部，并要求修好后再标记完成，用 agent 的语言（`protocol_language`）。重试次数是 frontmatter 的 `check_retries`，默认 2，即最多执行 3 次。
- **检查期间**：runner 照常读收件箱，你发的消息立即送进会话，这一轮接着跑。**停止**会杀掉正在执行的检查和它启动的进程（检查进程带 `AGENT_SYS_CHECK_RUN` 标记），这次尝试不记录，运行输出里显示为"随停止中断"而不是失败；会话此时空闲，不再发 interrupt。runner 意外退出时，server 结束它留下的检查进程。
- **追问不重跑**：已完成的任务再跑一轮，如果主报告没有变化（与 launch.json 里记下的本轮开始前的状态比较），就算回答追问：不跑检查，任务仍是已完成，不再标为未查看。
- **解析失败拒绝运行**：任务的 frontmatter 或 checks 解析不了时（扫描时发现），任何方式开始运行都会被拒绝（否则会不带检查、用默认的机器和预算运行）；错误写在任务的提醒里，待开始的任务等待原因是"prompt 有错误"，调度器不启动它，改正后才会开始。
- **终端接管**：claude 退出后，包装进程在 progress.md 已完成时执行一次检查（没有会话可以发回，不重试），结果打印在终端里并写进 `checks.json`，然后写结束记录。界面上的"结束接管"结束交互式 claude，最多等包装进程 5 s；检查还在跑就让它跑完（界面显示正在跑的检查，不再提供"结束接管"），结束记录写好后 Monitor 再推进任务。
- **检查用到的文件防改**：开始一轮时，launch.json 的 `toolchain` 记下检查工具链的 sha256：本包里检查会调用的模块（`ledger`、`perfcheck` 和检查命令里写出的 `agent_sys_v2.<模块>`，按 import 关系展开）、内置 skills 的 `scripts/`、项目 `skills/**/scripts/`，以及检查命令里直接写出路径、不在 `workspaces/` 下的文件。每次执行检查前重新计算并比对，有变化就在这次尝试的最前面记一条必定失败的检查（"the files the checks use are unchanged"），列出改了、加了、删了哪些文件；agent 收到的反馈里同样有这份清单。目的是防 agent 顺手把检查改松，不防恶意。agent_sys_v2 和项目要改的 Infera 仓库共用一个 checkout 时，那里的 `git checkout`/`pull` 也会触发它，所以报错里提示从单独的 checkout 或 wheel 安装。launch.json 里没有 `toolchain`（旧版本写的）时不比对。
- **口径快照**：同时把项目的台账口径（`ledger.primary`、`higher_is_better`、`x`、`x_higher_is_better`、`baseline_series`、`noise_pct`、`accept_pct`）快照进 launch.json 的 `ledger`，launch.md 的"验收依据"一节列出快照和检查文件的个数。launch.json 在 agent 能写的工作区里，所以 runner 用它启动时（agent 开始之前）读到的快照，经环境变量 `AGENT_SYS_RULES` 交给检查；检查还带 `PYTHONSAFEPATH=1`，项目目录下同名的 `agent_sys_v2*` 也算进检查文件（Python 3.10 不认这个变量，靠后者发现）。`ledger verify` 按快照判（§5.6）。
- **默认值**：类型骨架 `task_types/<类型>.md` 的 frontmatter 可以带默认的 `checks`（和 `review`）。在界面上新建任务时预填；agent 起草的草稿纳入托管时，frontmatter 没写的就取类型的默认值。

写法：

```yaml
checks:
  - test -s "$AGENT_SYS_WORKSPACE/report.md"
  - {name: results, run: '"$AGENT_SYS_PYTHON" -m agent_sys_v2.ledger verify "$AGENT_SYS_WORKSPACE/results"', timeout: 60}
check_retries: 2
```

### 5.5 调度

每 2 s，扫描完之后执行一次（只在 server 里跑：没有 server 就没有调度）。待开始的任务按以下顺序取第一个成立的等待原因：

1. 项目已归档
2. prompt 有错误（frontmatter 或 checks 解析不了，改正后才会开始；§5.4）
3. 阻塞类上游未完成
4. 等上游的子任务（`children` 依赖，§6.1）
5. 等你审核模板（§6.1）
6. 自动开始已关闭
7. **机器被占用**（写明占用者）
8. 项目名额已满
9. 全局名额已满

1–6 只看本项目，扫描时算出；7–9 跨项目，由调度器算出。没有等待原因的任务按等待时间从长到短启动，一次启动一个，每启动一个都重新扫描。运行中的任务占一个名额，在后台等待和终端接管期间也算；未托管的任务不占。

- **机器互斥**在 `start_run` 里检查，不只在调度器里：手动开始、发消息、继续、接管、追加预算都经过 `start_run`，在写锁下读所有项目的最新状态，机器被别的任务占用就拒绝；任务自己已占着机器时也检查（§3.2）。
- **手动开始**不看上游和名额，但仍检查预算、frontmatter 和机器。项目的模板还在审核中时，手动开始其中一个任务也就结束了审核。
- **自动开始失败**的任务进入待确认：预算不够时是"预算用完"，否则是"运行中断"并写明原因；不会自动重试。

### 5.6 台账

- **读哪些文件**：每个任务工作区 `results/` 顶层的 `*.json`（子目录放原始数据，不读），`schema` 以 `infera.result/` 开头的才算结果，其他 JSON 忽略。文件按 mtime 和大小缓存；Watcher 也只看 `results/*.json`，子目录里的原始数据变化不触发重新扫描。项目每次重新扫描时算一次（给看板头部和卡片的结果徽章），打开台账页时再算一次。
- **校验**：必填 `schema`、`label`、`metrics`（至少一个数字；NaN 和 Infinity 不算数字），`role`、`round`、`series`、`workload.conc`、`verdict` 的取值也要对。格式错误的文件不进表，列在台账底部，任务卡片上的结果徽章也标出来。
- **基线**：`ledger.baseline` 指定一个任务或一个结果文件时，取其中最新的、有效且带主指标的结果；没有指定时，取最新的、有效的 `role: baseline` 结果。指定的任务或文件没有这样的结果时，退回到后一种取法，台账返回 `pin_missing`（写的是哪个），台账页给出提醒。
- **Δ 和 gain**：Δ（`delta_pct`，界面上叫"台账 Δ"）是主指标相对基线的百分比；gain 是同一个数按 `higher_is_better` 换成"正数更好"，界面按 gain 着色；作废的结果和判定为噪声的结果没有 gain（null），不着色。有基线系列（`ledger.baseline_series`，没写时取基线结果所在的系列）时，带并发（`workload.conc`）的结果和基线系列在同一并发的**参照**比（多次取平均），参照没有这个并发就不给 Δ。
- **参照和复测**（`ledger.reference(table, series)`，后续的 A/B 门复用）：一个系列的参照只取封版这个版本的任务测的有效点：`v0` 取类型为 baseline 的任务，`vN` 取 `close/rN`。同一系列里别的任务测的点（实验的基线臂按 skill 记成基线系列的复测）是**复测**（行上 `repeat: true`）：照样画在曲线上（方块），但不进参照、不进前沿、不进工作点的值，所以参照不会随实验增多而漂移。不是版本名的系列、或者没有封版任务测过的系列，参照是它的全部有效点。
- **判定 Δ**：`verdict_delta_pct` 是判定里写的 `vs_baseline_pct`，相对这次实验自己的基线臂；和台账 Δ 分两列显示。判定所用的基线臂所属系列（`verdict.inputs.baseline_series`）不是当前基线系列时（例如封版后 `baseline_series` 从 v0 变成 v1），行上的 `judged_against` 写出它，界面标"相对 v0 判定"。
- **判定可复算**：带判定（decision、inputs 或作废）的结果，不论 role，台账都用 `ledger verify` 的同一套检查复算（不比较此刻的 project.yaml），结果放在 `verified`（无判定为 null）和 `unverified`（原因）里，界面标"判定未核实"并显示原因。复算按它读过的所有文件（结果文件、输入、原始输出、同系列同并发的结果、results/ 目录、任务的运行目录、project.yaml）的 mtime 和大小缓存。没有 `inputs` 的旧判定照常显示、标"未核实"，不算格式错误。
- **可计入最佳**：有效，并且是判定为接受的候选，或者是没有判定的基线（`role: reference` 是别人的数；带判定的基线不是 A/B）。噪声、需重跑、拒绝和还没判定的候选都不能当最佳。台账页、看板头部和卡片徽章都按这一条规则（`ledger.counts`）。
- **曲线**：结果带 `round`（轮次）和 `series`（曲线名：`v0`、`v1` 是版本，`r2-spf` 是一个实验，`ix:<条目>` 是对标）；`role: reference` 是别人的数（对标）。配置了 `ledger.x`（曲线的 X；`x_higher_is_better` 默认 true）时，每个系列是一条 Y–X 曲线：按 X 排序，标出前沿（同系列里没有别的点两个轴都不差）。`ledger.working_point` 是判 A/B 的工作点：`{conc: C}` 取该并发的点（多次取平均），`{x: X}` 取 X 不低于该值时的最大 Y（在前沿上相邻两点之间插值）；都只用参照点，不用复测。各系列在工作点的值相对基线系列的 Δ 和 gain 显示在台账页的系列表里。一个系列的点全部可计入最佳，这个系列才可计入最佳。
- **头部**：基线系列到达工作点时，头部（台账页和看板）按工作点比较系列：基线是基线系列在工作点的值，最佳是可计入最佳的系列里在工作点 gain 最大的那个。否则头部是基线结果和最佳结果：结果按并发和基线系列比较时（不同并发的值之间不可比），最佳是可计入最佳的结果里相对同并发基线 gain 最大的那个（都不比基线好时就是基线），没有"目前最佳"（`best_so_far` 为空）；否则最佳是可计入最佳的结果里主指标最好的那个，趋势图另画"目前最佳"阶梯线。
- **卡片徽章**：结果数、除基线外可计入最佳的结果里最好的 Δ、最近一个结果的判定、格式错误的文件数。
- **AgentX 转换**：`"$AGENT_SYS_PYTHON" -m agent_sys_v2.ledger agentx <agentx_conc*.json 或目录> --out "$AGENT_SYS_WORKSPACE/results" --series v0 --round 1 --role baseline` 把 InferenceX AgentX 的输出转成结果文件，指标名和算法与 InferenceX App 入库一致（交互性 = 1/p90 full-response ITL）；不覆盖已有的结果文件，原子写入；`--label` 和 `--series` 会成为文件名的一部分，含 `/` 时报错。`evidence` 记下来源：`agentx`（原始文件，在项目目录下时是相对项目目录的路径，否则是绝对路径）、`agentx_sha256`，以及同一运行目录里的 `runtime_env`（路径、sha256 和其中的变量，例如 `DURATION`）。只有带 `agentx` 和 `agentx_sha256` 的结果能作判定的输入；别的结果照常进台账，作参考。
- **当作检查**：`"$AGENT_SYS_PYTHON" -m agent_sys_v2.ledger check <目录或文件> [--require-verdict]`：有文件不对或一个结果都没有时退出码为 1 并列出原因；`--require-verdict` 要求每个候选都有判定或已作废。

**A/B 判定**（`verdict.py`；性能只以 AgentX 为依据）：
- **写判定**：skill infera-experiment 的 `ab_judge.py --write` 只接受 `--baseline-files`、`--candidate-files`，文件必须有 AgentX 来源，且原始文件的摘要和重新转换出的指标都对得上；手填的数字（`--baseline 12010 11950`）只打印试算，不写。指标、方向和 X 取 project.yaml 的 `ledger.primary`、`higher_is_better`、`ledger.x`；噪声带和接受线取 `ledger.noise_pct`、`accept_pct`（默认 2、3，从 `$AGENT_SYS_PROJECT` 或结果文件所在的项目读）：`--accept-pct` 只能提高接受线，噪声带不能在命令行改（更宽的噪声带也会把拒绝变成噪声，不算更严）。两臂在同一个并发，基线臂是 `ledger.baseline_series` 的复测、候选不是。所用文件所在任务的 `results/` 顶层里，同一并发的有效结果不论写的是什么系列和角色，都要参与（作废的除外；一个任务每个并发一个 A/B），否则拒绝；并发按原始输出算，改结果文件里的字段藏不住。判定只看数字：主指标增益超过接受线和噪声底、X 不劣于它自己的噪声底、中位数和均值方向一致才接受；`--mechanism` 是备注，不是条件。
- **`verdict.inputs`**：`metric`、`higher_is_better`、`x_metric`、`x_higher_is_better`、`baseline_series`、`baseline` 和 `candidate`（每份 `{file, agentx, agentx_sha256, value, x}`，路径相对项目目录）、`noise_pct`、`accept_pct`、`rules_from`（各自的出处，例如 `project.yaml` 或 `--accept-pct 4 (stricter than project.yaml: 3)`）。结果文件格式只做加法，仍是 `infera.result/v1`。
- **作废只有两类**（`verdict.void.kind`）：`failed_requests`，由 AgentX 的请求数和错误数算出失败比例（错误 /（成功 + 错误）），超过 10% 才能作废，不让 agent 填；`other`，写 `reason` 和存在的 `evidence` 文件。
- **`ledger verify <目录或文件>`**（experiment 类型的默认检查）：一份判定都没有（没有带 decision 或作废的结果）就失败。对每个带判定的结果（不论 role：把 role 改成 baseline 或 reference 藏不住判定，非候选带 decision 直接判为对不上），先做 `check --require-verdict` 的全部检查，再：要有 `verdict.inputs`；文件本身在候选输入里；每个输入的结果文件在、没作废、有 AgentX 来源，原始文件在、sha256 一致，**重新转换一遍**，指标和结果文件里的一致，`value`/`x` 和判定记下的一致，失败请求不超过 10%；两臂在同一个并发，所用文件所在任务里同一并发的有效结果都在输入里，作废的符合两类；指标、方向、基线系列和噪声带是快照的，接受线不比快照低；用同一个判定函数重算，`decision`、`vs_baseline_pct`、`noise_floor_pct` 一致。快照取 runner 交给检查的 `AGENT_SYS_RULES`，否则取正在执行检查的那一轮（`AGENT_SYS_CHECK_RUN`）或任务最近一轮的 launch.json；快照里缺的键（或没有快照）取 project.yaml；只有来自某一轮的快照才检查基线系列（台账用"相对 v0 判定"另行标出）。作为检查执行时还比较此刻 project.yaml 的口径和快照，本轮中被改过就失败。失败分两类：**判定和数据对不上**（the verdict does not match the data）和**这份判定无法复算**（the verdict cannot be recomputed：没有 AgentX 来源、原始文件没了、没有 inputs、文件读不了），每条按"问题、上下文（context）、怎么改（fix）"写。

`infera.result/v1`（必填：`schema`、`label`、`metrics`）：

```json
{
  "schema": "infera.result/v1",
  "label": "c64-spf",
  "time": "2026-10-24T10:00:00Z",
  "role": "candidate",
  "round": 1,
  "series": "r1-spf",
  "change": "--schedule-policy shortest-prefill-first (decode/prefill both)",
  "metrics": {"input_tput_per_gpu": 14120.5, "p90_intvty": 31.2, "p90_ttft": 2.1, "requests": 3120, "errors": 0},
  "workload": {"client": "agentx", "conc": 64, "duration_s": 3600, "seed": 42},
  "config": {"kit": "atom_4p4d_glm5.2", "image": "…", "extra_args": "…"},
  "verdict": {"status": "valid", "decision": "accept", "vs_baseline_pct": 15.4, "noise_floor_pct": 2.0,
              "notes": "+15.40% past the 3% bar and the floor 2.00%", "mechanism": "[M] queue-req ↓ 80%",
              "inputs": {"metric": "input_tput_per_gpu", "higher_is_better": true, "x_metric": "p90_intvty",
                         "x_higher_is_better": true, "baseline_series": "v0",
                         "baseline": [{"file": "workspaces/experiment/r1-spf/results/r1-spf-base-c64.json",
                                       "agentx": "workspaces/experiment/r1-spf/log/agentx/base/agentx_conc64.json",
                                       "agentx_sha256": "…", "value": 12236.1, "x": 31.0}],
                         "candidate": [{"file": "…/r1-spf-c64.json", "agentx": "…", "agentx_sha256": "…", "value": 14120.5, "x": 31.2}],
                         "noise_pct": 2.0, "accept_pct": 3.0,
                         "rules_from": {"noise_pct": "project.yaml", "accept_pct": "project.yaml"}}},
  "evidence": {"agentx": "workspaces/experiment/r1-spf/log/agentx/spf/agentx_conc64.json", "agentx_sha256": "…",
               "runtime_env": {"file": "…/runtime.env", "sha256": "…", "vars": {"DURATION": "3600", "…": "…"}}}
}
```

`role`：`baseline` / `candidate`（默认）/ `reference`；`verdict.status`：`valid`（默认）/ `void`；`verdict.decision`：`accept` / `reject` / `noise` / `rerun`；作废时 `verdict.void` 是 `{"kind": "failed_requests", "failed_pct": 12.3}` 或 `{"kind": "other", "reason": "…", "evidence": "<文件>"}`。

### 5.7 skills 注入

- system.md 有一段"skills 索引"，一行一个：名字、描述、SKILL.md 的绝对路径。agent 用得上时自己去读。用法和经验索引相同，增加的上下文很小。
- 注入哪些：`project.yaml` 的 `skills` 列表里的内置 skills，按列表的顺序（不写 `skills` 则注入全部内置 skills），再加项目自己 `skills/` 目录下的全部（与内置的同名时取项目自己的）。`skills` 里写了不存在的名字，launch.md 里会提醒。
- SKILL.md 的 frontmatter 和 Claude Code 原生 skill 的格式一致（`name`、`description`）；没有 description 的不算 skill。
- system.md 各段的顺序：agent_sys 约定（`protocol_language` 选中文版或英文版）、全局规则、项目规则、skills 索引、项目经验索引、全局经验索引。开头写明项目目录、`AGENT_SYS_HOME`、本任务的 prompt 和工作区。

### 5.8 API

路径都在 `/api` 下，任务的路径前缀是 `/api/projects/{p}/tasks/{类型}/{名字}`（下表写作 `…`）。请求体是 JSON，字段类型不对返回 400；拒绝的请求返回 `{"error": <给用户看的文字>}`，状态码 400 输入有误、403 跨站或 Host 不对（§5.9）、404 不存在、409 当前不允许（或冲突、已存在）、422 文件无法解析。文件路径都相对于 `AGENT_SYS_HOME`。

| 方法与路径 | 说明 |
|---|---|
| `GET /api/home` | 首页：未归档项目的待确认、待审核、未查看、运行中、待开始数，今日花费；项目和已归档项目；被占用的机器（每台的全部占用者 `holders`）；模板列表；新项目所在的目录 |
| `POST /api/resume` | 继续全部中断任务：`{tasks: [{project, task}]}`，只继续其中运行中断的 |
| `POST /api/mark_read` | 全部标为已查看 |
| `POST /api/projects` | 新建项目 `{name, template, title}` |
| `POST /api/projects/{p}/types` | 新建类型 |
| `PUT /api/projects/{p}/lanes` | 泳道顺序 |
| `POST /api/projects/{p}/archive`、`…/unarchive` | 归档、取消归档 |
| `POST /api/projects/{p}/start` | 开始运行：结束模板审核 |
| `GET /api/projects/{p}/board` | 看板：项目（含目标与台账摘要）、类型、任务、依赖边（含是否阻塞、`children` 的完成数）、机器表（每台机器的 `holders` 列表，空闲时为空）、名额 |
| `GET /api/projects/{p}/ledger` | 台账：设置、主指标、基线（`pin_missing`：钉住的基线没有可用结果）、最佳、行、系列、格式错误的文件 |
| `POST /api/projects/{p}/tasks` | 新建任务（排队或存为草稿） |
| `POST /api/projects/{p}/tasks/preview` | 新建任务表单的预览：prompt 和所选上游给出的依赖，名字能不能用 |
| `GET …` | 任务详情：上下游、草稿、本轮的问题、最近一轮的检查、运行列表等。prompt 的错误（frontmatter、checks 解析不了）在 `warnings` 里 |
| `PUT …` | 修改任务配置（frontmatter 字段、prompt）；`send_to_agent` 时把新 prompt 告诉已运行过的任务，告诉不了时照样保存并在 `notify_error` 里说明 |
| `POST …/rename` | 改名或改类型 |
| `POST …/actions/{action}` | `start`、`stop`、`resume`、`takeover`、`end_takeover`、`add_budget`、`cancel`、`join`、`join_children`（批量纳入子任务草稿）、`approve`、`mark_read`、`delete`（移进 `.trash/`） |
| `POST …/messages` | 你的消息：送进运行中的会话，或开始新一轮 |
| `GET/POST …/questions` | questions.md 表单：读、提交回答 |
| `GET …/runs/{run}/events?after=&before=&tail=` | 精简过的事件（长内容截断）；`GET …/runs/{run}/events/{seq}` 取一条全文 |
| `GET …/runs/{run}/launch` | 这一轮的 launch.md |
| `GET …/files?dir=`、`GET/PUT …/file?path=`、`GET …/raw?path=` | 工作区文件：列目录、读、写、原样下载 |
| `GET/PUT /api/rules` | 规则页（`?project=` 带上项目的） |
| `GET /api/knowledge`、`POST /api/knowledge/candidates/{adopt,to_rule,discard}`、`PUT /api/knowledge/entries`、`POST /api/knowledge/entries/delete` | 经验：候选和经验库、采纳 / 改写为规则 / 丢弃、编辑和删除条目 |
| `GET/PUT /api/settings` | 设置（全局，或 `project` 指定的项目） |
| `GET /api/models` | 表单里可选的模型，以及别名、完整 id 在 agent 用的设置下（§5.9）指向什么 |
| `GET /api/events` | SSE：`changed {project, task}`（task 为 null 表示项目本身）、`run {project, task, run, seq}`（运行有新事件） |

### 5.9 安全与边界

- **只监听 Unix socket**，默认 `/tmp/agent_sys-<uid>/agent_sys.sock`，目录权限 0700，只有你能连。浏览器通过 `ssh -L` 转发访问。
- **Host 校验**：转发之后，浏览器把 server 当成 localhost 访问，你电脑上的任何网页都能向它发请求。所以请求的 Host（去掉端口）必须是 `localhost`、`127.0.0.1`、`[::1]`，或环境变量 `AGENT_SYS_HOSTS`（逗号分隔）里的名字，否则返回 403（防 DNS rebinding）。
- **Origin 校验**：GET / HEAD / OPTIONS 以外的请求必须来自 server 自己的页面：有 `Origin` 时它的 host:port 要等于 Host；没有 `Origin` 时 `Sec-Fetch-Site` 要是 `same-origin` 或 `none`。两个头都没有（curl 之类）的请求放行。
- **CSP**：所有响应带 `default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'`：脚本只从本站加载，组件可以用内联样式，图片可以是 data: 和 blob:，别的网站不能把界面嵌进 frame。agent 写的 Markdown 用 marked + DOMPurify 渲染，去掉脚本、表单、控件和样式。
- **`/raw`** 一律带 `Content-Security-Policy: sandbox` 和 `X-Content-Type-Options: nosniff`：不管文件是什么后缀，都在没有脚本的不透明源里打开，不能调用 agent_sys 的 API。
- **工作区文件**：路径相对于工作区，不接受绝对路径和 `..`。读取跟随符号链接，可以读到工作区以外（工作区常常链接到别处的数据，而只有你能连上 server）。写入先解析符号链接：真实路径必须在工作区内，否则拒绝；工作区内的符号链接保持是链接，写的是它指向的文件，文件权限不变；`.agent_sys/` 不能通过界面修改。保存时带上开始编辑时的 mtime，文件期间被改过就拒绝（409）。
- **改名和删除**会移动工作区，运行中的任务、刚纳入托管（终端会话可能还没结束）的任务会被拒绝。
- agent 以 `bypassPermissions` 运行，靠规则、检查和人的审核约束，不做 OS 隔离：agent 要操作集群、容器和仓库，隔离起来就做不了这些事。
- server 和 runner 去掉启动它们的 shell 里的 `CLAUDE*`、`ANTHROPIC*`、`IS_SANDBOX` 变量，避免在 Claude Code 会话里启动时覆盖用户配置。发给 agent 的消息里不带任何凭据。
- **agent 的 Claude Code 配置与你的隔离**（`claude_config.py`，`agent_sys.yaml` 的 `claude_config`，默认 `isolated`）。你的 `~/.claude` 里有你的 MCP server（`~/.claude.json`）、插件（`enabledPlugins`，以及 env 块里的 `CLAUDE_CODE_PLUGIN_DIRS`）、CLAUDE.md、skills 和 hooks。agent 以 `bypassPermissions` 运行，继承它们就等于不经确认地拿到你的外部系统写权限（实测一次运行带了 32 个 Atlassian MCP 工具：建 Jira issue、建和改 Confluence 页面），你的插件的 hook 也在 agent 会话里生效。所以：
  - runner 和终端接管都以 `CLAUDE_CONFIG_DIR=$AGENT_SYS_HOME/claude` 启动 claude，加 `--strict-mcp-config`，setting sources 仍只有 user（即这个目录）；
  - 这个目录的 `settings.json` 每次启动按 `~/.claude/settings.json` 重新生成（文件权限 0600，目录 0700），只带下面两张白名单里的东西；launch.json 和 launch.md 只记键名，不记值；
  - `projects/` 是指向 `~/.claude/projects` 的符号链接：resume、终端接管、`session_cost`、隔离之前的会话都照旧在那里找。链接幂等；那里已有非空的真目录或指向别处的链接时不删、不改，拒绝开始并说明（409）；
  - `.claude.json`（Claude Code 自己的状态）缺什么补什么：`hasCompletedOnboarding`、转发的 API key 已确认（`customApiKeyResponses`，存 key 的后 20 个字符，和 Claude Code 一样）、项目目录已信任。全新目录里交互式 claude 依次弹：选主题、确认 API key、登录、信任目录，再加上没有 `skipDangerousModePermissionPrompt` 时的 bypass 警告，终端接管会卡在第一个上；补齐后都不弹（实测）。交互式 claude 仍会把官方插件市场的目录 clone 进来，但不启用其中的插件；
  - OAuth 登录：`~/.claude/.credentials.json` 里有 Claude 的登录（`claudeAiOauth`）时，agent 的环境加 `CLAUDE_SECURESTORAGE_CONFIG_DIR=~/.claude`，原地读写（实测全新配置目录加这个变量后 `claude auth status` 是 claude.ai 登录）。不软链：软链读得到，但 Claude Code 写凭据（刷新 token）时不跟随软链（`O_NOFOLLOW`）；拷贝则在第一次刷新后过期。文件里只有 MCP server 的 OAuth token（`mcpOAuth`，本机就是这样：Atlassian 的 token）时不设，agent 拿不到它们；
  - 项目需要 MCP server 或插件时，在 `agent.extra_args` 里写 `--mcp-config <文件>`（只能一个，SDK 只传一个文件；多个 server 写进同一个文件）、`--plugin-dir <目录>`（可多个）。launch.md 的"Claude Code 配置"一节列出显式加的这些；
  - `claude_config: inherit` 退回旧行为：用你的 `~/.claude`，不加 `--strict-mcp-config`。设置页可以切换；launch.json 里没有 `claude` 字段的旧运行也按旧行为启动。

  **env 白名单**（`FORWARD_ENV`）：登录和模型路由（`ANTHROPIC_*`；`CLAUDE_CODE_USE_BEDROCK/VERTEX/FOUNDRY`、`CLAUDE_CODE_SKIP_*_AUTH`、`AWS_*`、`GOOGLE_*`、`CLOUD_ML_REGION`、`VERTEX_REGION_*`；`CLAUDE_CODE_OAUTH_TOKEN`、`CLAUDE_CODE_API_KEY_HELPER_TTL_MS`、`CLAUDE_CODE_CLIENT_CERT/KEY/KEY_PASSPHRASE`），网络（`HTTP(S)_PROXY`、`NO_PROXY` 及小写、`NODE_EXTRA_CA_CERTS`、`SSL_CERT_FILE`），网关和模型要的开关（`CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC`、`CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING`、`CLAUDE_CODE_MAX_OUTPUT_TOKENS`、`CLAUDE_CODE_SUBAGENT_MODEL`、`MAX_THINKING_TOKENS`、`API_TIMEOUT_MS`、`DISABLE_TELEMETRY` 等），`IS_SANDBOX`（root 下跑 bypassPermissions 要它）。明确不转发：`CLAUDE_CODE_PLUGIN_DIRS`、`CLAUDE_CONFIG_DIR`、`CLAUDE_CODE_TMPDIR`、`PATH`、`TMPDIR`；不在白名单里的（如 `CLAUDE_CODE_DANGEROUSLY_SKIP_PERMISSIONS`，runner 自己定权限模式）一律不转发，launch.md 列出名字。agent_sys 自己的环境里已有的键不写进文件，agent 继承环境里的值：settings 的 env 块会盖过进程环境（实测），写进去就成了文件优先。

  **settings 键白名单**（`CARRIED`，逐个过了本机 settings.json 的顶层键）：

  | 键 | 带不带 | 理由 |
  |---|---|---|
  | `env` | 按 env 白名单 | 见上 |
  | `modelOverrides` | 带 | 把完整 id（`claude-opus-5-5` 等）映射到网关的名字。不带时按完整 id 选模型，请求里发的是原 id（实测），网关不认就路由不到 |
  | `modelSettings`、`model` | 带 | 按模型的设置（如 effort）；任务没写模型时的默认模型 |
  | `apiKeyHelper`、`awsAuthRefresh`、`awsCredentialExport` | 带 | 登录 |
  | `skipDangerousModePermissionPrompt` | 固定为 true | 不论你怎么设：终端接管以 bypassPermissions 启动交互式 claude，否则先弹危险模式确认 |
  | `skipWorkflowUsageWarning` | 带 | 开了 ultracode 的终端接管不先弹用量提醒 |
  | `autoCompactEnabled` | 带 | 决定长会话上下文满了以后怎么办，会改变 agent 的行为；隔离之前 agent 一直按你的设置跑，保持不变 |
  | `permissions` | 只带 `deny` | deny 只会收窄 agent 能做的事；`defaultMode` 由 runner 定（bypassPermissions），`allow`、`ask`、`additionalDirectories` 在这个模式下没有意义 |
  | `enabledPlugins`、`extraKnownMarketplaces`、`hooks` | 不带 | 插件和 hook 正是要隔开的东西 |
  | `theme`、`editorMode`、`tui`、`verbose`、`statusLine`、`outputStyle`、`askUserQuestionTimeout` | 不带 | 只影响你自己的界面 |
  | `cleanupPeriodDays` | 不带 | runner 用 `--settings` 设 365 天 |
  | 其余 | 不带 | launch.md 列出名字；需要时加进 `CARRIED` |

  `launch.models()`（表单里"别名和完整 id 指向什么"）读的是同一份：隔离时是生成给 agent 的设置，env 的值环境优先；`inherit` 时是你的 settings.json。

### 5.10 环境变量

| 变量 | 谁读 | 作用 |
|---|---|---|
| `AGENT_SYS_HOME` | `agent-sys-v2` 命令 | home 目录：`--home` 优先，其次这个变量，再次 `~/agent_sys_home` |
| `AGENT_SYS_TMUX` | server、runner | tmux server 的名字（`tmux -L`），默认 `agent_sys`；在正式 server 旁边再开一个（测试、演示）时换一个 |
| `AGENT_SYS_HOSTS` | server | Host 校验额外接受的名字，逗号分隔（§5.9） |
| `AGENT_SYS_PYTHON`、`AGENT_SYS_PROJECT`、`AGENT_SYS_TASK`、`AGENT_SYS_WORKSPACE`、`AGENT_SYS_MACHINES`、`AGENT_SYS_SKILLS` | agent、检查 | 运行 agent_sys 的 Python、项目目录、任务 id、工作区、机器（逗号分隔）、内置 skills 的目录（§5.4） |
| `AGENT_SYS_CHECK_RUN` | agent_sys 自己 | 检查进程的标记（值是 run 目录解析符号链接后的路径），停止时据此找到并结束检查；`ledger verify` 据此找到这一轮的口径快照 |
| `AGENT_SYS_RULES` | 检查 | runner 交给检查的这一轮台账口径快照（JSON），`ledger verify` 优先用它 |

隔离时（§5.9）agent 的环境里有 `CLAUDE_CONFIG_DIR=$AGENT_SYS_HOME/claude`，有 Claude 的 OAuth 登录时还有 `CLAUDE_SECURESTORAGE_CONFIG_DIR=~/.claude`。另外，agent 的环境里还加了 `BASH_MAX_TIMEOUT_MS=86400000`（后台命令最长 24 h）、`CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1`（内存紧张时不回收后台命令）、`CLAUDE_CODE_EMIT_SESSION_STATE_EVENTS=1`（runner 靠会话状态事件判断一轮结束）、`CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`（不写用户的 auto memory）。

## 6. Infera 性能工作包

### 6.1 模板 `infera-perf`

| 类型 | 用途 | 默认检查 |
|---|---|---|
| `intake` | 每轮开头的问卷和 brief（`review: true`）；可写项目的 `rules/project.md`、`project.yaml`、集群 skill | report.md 非空；`perfcheck intake` |
| `survey` | 对标调研：InferenceX 参考曲线（`role: reference`）、配置差异表、合规清单；不占 GPU | report.md 非空；`perfcheck survey`；有 results 时 `ledger check` |
| `baseline` | 部署、过质量门、测 v0 曲线（`role: baseline`），噪声和快速验证标定，栈清单 | report.md、`ledger check`、calibration.json、stack.json |
| `profile` | 从便宜的层做起的瓶颈分析，产出 `candidates.yaml`（`review: true`；skill infera-profile；默认不用 ultracode） | report.md、candidates.yaml、profile.json |
| `plan` | 候选排序、预算和机器分配，起草 experiment 子任务和 close（`review: true`） | report.md 非空；`perfcheck plan` |
| `experiment` | 一个大杠杆或一组同层小杠杆：快速验证迭代，工作点单点 A/B（AgentX），写带判定的结果 | report.md 非空；`ledger verify` |
| `close` | 集成、整条曲线、封版、复盘，起草下一轮 intake；可写 `project.yaml` 的 version 和基线系列 | report.md 非空；有 results 时 `ledger check`；`perfcheck close` |
| `develop` | 代码改动：补丁、router、kernel，出 PR 或 overlay | report.md 非空 |
| `review` | 评审一个 PR、分支或结论（`review: true`） | — |

`perfcheck`（`"$AGENT_SYS_PYTHON" -m agent_sys_v2.perfcheck <类型> "$AGENT_SYS_TASK"`）是这个模板的字段检查，在项目目录下运行，每项打印一行 ok / FAIL，有一项失败就退出码为 1；轮次取任务名里的数字（`intake/r2`）。
- intake：project.yaml 能解析，有目标、机器、轮次、台账口径（primary、x、working_point）；`rules/project.md` 没有剩下占位；问过问卷；本轮要上机的任务声明了机器；第 2 轮起本轮的 plan 已起草（报告写明项目结束时除外）。
- survey：results/ 里有对标结果（`role: reference`），或报告写明没有公开对标。
- plan：每个子任务声明了项目里有的机器，写了 `expect`（预期 Δ、快速验证级别 L0–L5、证伪条件）；有一个任务以 `children` 依赖等它的子任务，并声明了机器。
- close：`stack.json` 写明版本和是否封版；封版时这个系列在 2 个以上并发有点，并且是 project.yaml 的 version 和台账的基线系列；下一轮的 intake 已起草，并排在本任务之后。

一轮的做法、文件格式和问卷在模板自带的项目 skill `infera-round` 里。

两个通用机制让模板跑得起来：
- **模板自带任务**：`templates/<模板>/tasks/<类型>/<名字>.md` 在建项目时原样复制并排队（`skills/` 也复制，成为项目自己的 skill）。它们不会自己开始：project.yaml 写入 `template_review: true`，调度器不启动这个项目的任务，没有上游的任务等待原因是"等待你审核模板"。你审核后在看板上点"开始运行"（`POST /api/projects/<p>/start`），或手动开始其中一个任务，这一行就删掉，之后和普通项目一样按依赖自动推进。
- **`children` 依赖**：`{task: X, kind: children}` 要等 X 完成，并且 `parent: X` 的任务全部完成或取消（草稿也算没完成；等待者自己和它的下游不计入）。看板上画成青色点划线，线的提示里写出子任务的完成数，等待原因写"等待 X 的子任务：…"，agent 的首条消息列出 X 的子任务和它们的报告。X 的"全部纳入托管"纳入 X 起草的所有草稿：`parent: X` 的，以及直接依赖 X 的（评审覆盖除外）。草稿纳入托管时，frontmatter 没写的 `review` 和 `checks` 取类型的默认值。

项目规则 `rules/project.md` 写这个项目的背景（模型、引擎、kit、节点、负载、目标口径、工作点、SLA），以及实验纪律：
- 先诊断再选杠杆；一个子任务一个大杠杆，或一组同层、同验证方法的小杠杆，改动前写清预期机理、幅度和证伪条件；
- 以实际运行的配置为准，证明杠杆生效了再测；
- 证据标 `[M]` 实测或 `[I]` 推断；
- 判定只看 AgentX 结果（性能以 AgentX 为准，精度以 GSM8K 为准）：ΔY 超过接受线（`ledger.accept_pct`，默认 3%）和噪声底、ΔX 不劣于噪声、中位数和均值方向一致才接受，噪声底内（`ledger.noise_pct`，默认 2%）算噪声，介于两者之间加跑；机理指标、fast 模式、dstep 用于筛选和解释，不是接受条件；口径只由用户改，命令行只能提高接受线；用 X 换 Y 的改动在等 X 下比较（工作点 Δ）；
- 作废只有两类：失败请求超过 10%（从 AgentX 计数算）；其他原因（别人的负载、HSA out-of-resources、被抢占……）写原因和证据文件；
- 不动别人的容器和 GPU；长任务在远端以自包含的执行链运行，并带守护。

### 6.2 内置 skills

| skill | 内容 |
|---|---|
| `infera-deploy` | kit 的启停；VRAM 等待和外来容器检查；生效配置审计（`docker inspect`、`/get_server_info`），找出没被转发的 knob；守护规则 |
| `infera-bench` | AgentX 全量和 fast 模式、固定 ISL/OSL 负载；每个点的固定产物；写 `infera.result/v1`；测量陷阱 |
| `infera-quality-gate` | 拒绝在 forced acceptance 下跑；smoke、GSM8K、乱码和长上下文探针、MTP 真实接受、RDMA 不回退 TCP、PD 交接 |
| `infera-profile` | 一轮的 profile：从便宜的层做起（时间账、资源利用），只对瓶颈层做 kernel 或 CPU；给 plan 的 `candidates.yaml`；各引擎的 playbook |
| `infera-diagnose` | 瓶颈分类（prefill 饱和、decode 步长、KV 容量、传输、CPU 路径、JIT、正确性）；每一类的观测手段和候选杠杆 |
| `infera-experiment` | 单变量 A/B 协议、两类作废和判定规则、`scripts/ab_judge.py`（只用 AgentX 结果；口径取 project.yaml、命令行只能提高接受线；噪声底；噪声不会判成接受；同时看 X；`--write` 把判定和 `inputs` 原子写入每个候选文件）、`ledger verify` 自查 |
| `infera-hygiene` | 共享节点卫生：不删别人的容器，NFS、TMPDIR、HOME 陷阱，VRAM 释放，抢占 |

skills 的原则：
- 确定性的事交给脚本，判断交给 agent；
- 每个 skill 都写明"怎么证明它真的生效了"；
- 不复制仓库里已有的脚本，而是指过去（例如 `repos/Infera/examples/<kit>/engine/up.sh`）。

## 7. 可观测与可控清单

| 你想知道或做的事 | 在哪里 |
|---|---|
| 现在哪些事需要我 | 首页待确认（含验收未通过、闲置占用）；标签页标题上的数字 |
| 某个 agent 正在干什么 | 卡片上的"正在执行：<命令>"；运行输出实时流 |
| 它在等什么 | 运行中 · 等待中：后台命令的描述、已等多久；待开始的卡片写明等待原因 |
| 它拿到了哪些规则、skills、经验 | 运行输出顶部"本次注入内容"（launch.md） |
| 它带了我的哪些 Claude 配置 | launch.md 的"Claude Code 配置"：转发和没转发的 env 键、带和没带的 settings 键、显式加的 MCP 和插件 |
| 花了多少 | 卡片和抽屉里的花费/预算；首页今日花费 |
| 哪台机器被谁占着、是否闲置 | 看板机器条；首页机器卡片 |
| 优化走到哪了 | 台账：各系列的 Y–X 曲线和工作点 Δ，基线和最佳，每个结果的判定 |
| 结论可信吗 | 检查 ✓/✗；结果的判定和 `[M]/[I]` 证据；报告审核 |
| 让 agent 改方向 | 底部输入框（运行中也可以） |
| 立刻停下 | 停止（打断当前命令，检查也一起停） |
| 亲自上手 | 在终端中接管（同一会话），退出后自动接回 |
| 控制花费 | 每个任务的预算；追加预算 |
| 批准 agent 起草的计划 | 父任务抽屉：全部纳入托管 |
