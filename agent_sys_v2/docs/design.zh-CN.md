# agent_sys v2 设计：产品、前端与后端

本文写 agent_sys v2 的产品定位、架构和规则；安装、运行和开发见 [README](../README.zh-CN.md)。

约定：`<p>` 指项目名；除特别说明外，路径都相对于 `AGENT_SYS_HOME`；"§x.y"指本文的章节，代码注释里的"design §x.y"也指这里。

| 想了解 | 看 |
|---|---|
| 它是什么、和 v1 有什么不同 | §0、§1 |
| 一轮性能优化怎么推进 | §2、§6 |
| 任务、依赖、状态、机器的规则 | §3 |
| 网页上有什么、怎么操作 | §4、§7 |
| 进程、文件、配置、模块 | §5.1–§5.3 |
| agent 说"完成"之后发生什么 | §5.4 |
| 什么时候开始、开始前查什么、上游更新了怎么办 | §5.5 |
| 结果怎么记、A/B 怎么判、怎么复算 | §5.6 |
| 注入哪些 skills | §5.7 |
| API、安全边界、环境变量 | §5.8–§5.10 |
| 怎么发现卡住的运行 | §7.1 |

## 0. 一页概览

agent_sys v2 是 Infera 的**性能工程 agent 工作台**：一名工程师在网页上指挥一组 Claude Code agent，在 GPU 集群上找出并消除 Infera 部署的瓶颈。瓶颈可能在任何一层：拓扑与并行、引擎参数、CPU 路径、通信、KV 传输、kernel。每个结论都要有实测证据，每个 agent 都看得见、管得住。

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
| 工作流 | 事先声明的静态 DAG（YAML，5 类对象） | 一个任务就是一份 prompt。任务图在工作中长出来：agent 起草子任务，人批准 |
| agent 的角色 | 照着 brief 执行步骤（"do not invent a method"） | 自己决定查什么、试什么；skills 提供可靠的工具和方法 |
| 质量保证 | handoff schema + validator，管的是格式 | 实测结果台账 + A/B 判定规则 + 验收检查，管的是结论 |
| 可观测 | 状态变化时打一行；agent 的消息丢弃 | 每个 agent 的实时输出、花费、当前命令、后台等待、机器占用、实验曲线、疑似卡住 |
| 可控 | 只能 Ctrl-C 结束整个 run | 对单个任务发消息、停止、终端接管、追加预算、审核、回答问题 |

## 1. 定位与原则

**定位**：一名工程师加一群 agent 的工作台，人只做决策和验收。它不是给人用的项目管理工具，也不是自动驾驶的优化器。

**原则**：

1. **用户体验优先。** 需要人处理的事集中在一个页面，一行一件，配好主按钮。新建任务就是写一段话，不用先学一套 YAML 概念。界面和文件始终一致：不开界面，照同样的约定也能工作。
2. **信任 agent，核验证据。** 不规定 agent 的步骤，只规定证据的标准：结果格式、A/B 判定规则和验收检查。检查由 harness 执行，不由 agent 自己执行。两条经验：昂贵的门前先放便宜的检查；有意思的失败不是崩溃。
3. **可观测。** 运行的每一帧都落盘（`events.jsonl`），界面实时展示。注入了什么、花了多少、在等什么，都写在 `launch.md` 和卡片上。
4. **可控。** 任何一个 agent 都能单独发消息、停止、接管、加预算、审核。机器占用由调度器管，不靠口头约定。
5. **简单。** 文件就是状态，server 随时可以重启。一种进程模型，一套状态词汇。加新功能前先问：它能让产出更可信吗？能省人的时间吗？

## 2. 用户与核心循环

**用户**：
- **性能工程师**（主要用户）：给定模型 × 硬件 × 引擎 × 拓扑 × 负载，把 goodput 提上去。节点稀缺，希望几个实验并行；不想守着屏幕，早上来看到的是结果，而不是一台闲置了一夜的机器。
- **评审者**：看结论和证据，决定哪些改动进 overlay 补丁或 PR。
- **团队**：skills、经验库和基线在项目之间共享。

**核心循环**（模板 `infera-perf`，按轮推进；第 1 轮的前 5 个任务在建项目时由模板预置）：

```
 ① intake/rN        预填问卷 → 你回答 → 写项目的规则、目标、机器、工作点 → brief（审核）
        │
        ├──────────────────────────┐
 ② baseline/r1                     survey/rN
    v0 全量 GSM8K、v0 曲线、         对标曲线进台账（reference）、
    工作点同节点复测（噪声）          配置差异表；不占 GPU
        │                          │
 ③ profile/rN                      │
    从便宜的层做起：时间账 → 资源利用 → │
    只对瓶颈层做 kernel / CPU →       │
    candidates.yaml（审核）           │
        ├──────────────────────────┘
 ④ plan/rN          候选排序、GPU 预算和机器分配（审核）→ 起草 experiment 子任务和 close/rN
        │
 ⑤ experiment × N   一个大杠杆或一组同层小杠杆：快速验证迭代 → 工作点单点 A/B → 判定；按机器并行
        │
 ⑥ close/rN         以 children 依赖等 plan/rN 的子任务全部完成或取消；集成 → 整条曲线 →
                    不劣于 v(N−1) 时封版 vN → 复盘 → 起草 intake/r(N+1)（带建议的增量问卷）
```

第 1 轮里，profile 只等 baseline，plan 等 profile 和 survey。从第 2 轮起，本轮的任务由 intake/rN 按问卷回答起草（survey、plan，需要时 profile 和 baseline）；上一轮 close 测的集成曲线就是这一轮的基线，通常不再重做 baseline。一轮的任务图、各任务的产出和文件格式写在模板自带的项目 skill `infera-round` 里（§6.1）。

和固定流水线的区别：③决定④⑤做什么，⑤⑥的结果又决定下一轮做什么。kernel 优化只是③的一种结论，和 tokenizer workers、调度策略、P:D 比例、Mooncake 参数地位相同，先做哪个由②③的证据决定。

## 3. 概念模型

一共 7 个概念，每个都落在文件上：

| 概念 | 是什么 | 文件 |
|---|---|---|
| **项目**（project） | 一次优化活动：目标、机器、台账配置、规则、经验 | `projects/<p>/project.yaml`、`rules/`、`knowledge/` |
| **任务**（task） | 一份 prompt 加一个工作区；任务类型就是看板上的泳道 | `tasks/<类型>/<名字>.md`、`workspaces/<类型>/<名字>/` |
| **运行**（run） | 任务会话的一轮：启动参数、注入内容、全部事件、结束记录 | `workspaces/…/.agent_sys/runs/<run>/` |
| **检查**（check） | agent 报告完成后由 harness 执行的验收命令 | 任务 frontmatter 的 `checks`；结果在 `runs/<run>/checks.json` |
| **结果**（result） | 一个测量点，统一格式 | `workspaces/…/results/*.json`（`schema: infera.result/v1`） |
| **机器**（machine） | 项目可用的节点；任务声明要用哪些，调度器保证互斥 | `project.yaml` 的 `machines`；任务 frontmatter 的 `machines` |
| **skill** | 可复用的操作手册和脚本，例如部署、测量、分诊、A/B | `src/agent_sys_v2/skills/<名字>/SKILL.md`；项目自己的放 `projects/<p>/skills/` |

另有两个约定：
- **规则**：全局和项目的 `rules/*.md`，每次运行都注入。
- **经验**：任务把候选写在工作区的 `lessons/`，人采纳后进经验库；全局和项目经验库的 `INDEX.md` 每次运行都注入。

**任务托管**：系统管理状态、负责启动的任务叫**托管任务**，状态记在 `.agent_sys/state.json`。没有托管的任务（例如 agent 起草的草稿、在别处手工跑的任务）叫**未托管任务**，状态从 `progress.md` 推导（§3.1）。"纳入托管"就是把未托管任务交给系统。

**依赖**

任务之间的边有 5 种：

| 种类 | 含义 | 阻塞下游、参与环检测 |
|---|---|---|
| `input` | 产出依赖：要用上游的产出 | 是 |
| `code` | 代码基线：在上游的代码上继续改 | 是 |
| `children` | 等上游完成，并且等上游的全部子任务完成或取消（§6.1） | 是 |
| `review` | 评审覆盖：评审上游的结论 | 否 |
| `subtask` | 父子关系 | 否 |

边有三处来源，下面按优先级从低到高列出；同一对任务以优先级高的来源为准：
1. **prompt 里写到的其他任务**：`tasks/<类型>/<名字>`、`workspaces/<类型>/<名字>`（前面可以带 `projects/<本项目>/`；指向别的项目的路径不算）或 `@<类型>/<名字>`，记为 `input`；如果写到的是本任务的 parent，记为 `subtask`。
2. **frontmatter 的 `parent`**：记为 `subtask`。
3. **frontmatter 的 `depends_on`**：写 `类型/名字`，或 `{task, kind, label}`（kind 缺省为 `input`）。同一个上游写了多次时，后写的为准；kind 不认识的条目不成边，只在记录提醒和开始前检查里报出。

依赖自己的边一律忽略；在界面上这样写会被拒绝。

**环**：通过界面新建、修改或改名任务时，如果这次写入使任务首次出现在阻塞类依赖构成的环上，写入被拒绝（任务原本就在环上时不再拦）。直接改文件形成的环，环上每个任务都有记录提醒，开始前检查也会报错（§5.5）。

### 3.1 任务状态

每个状态都同时用形状、文字、颜色三种方式表示。

| 状态 | 含义 |
|---|---|
| 草稿 draft | 有 prompt，未托管，还没有 progress.md（agent 起草的子任务都在这里） |
| 待开始 queued | 卡片写明在等什么：项目已归档、prompt 有错误、上游未完成、等上游的子任务、等你审核模板、自动开始已关闭、**等待机器**（写明被谁占着）、名额已满（§5.5） |
| 运行中 running | 分三种：执行中、等待中（后台命令在等作业，不花 token）、终端接管。执行中还会按事件判断是否**疑似卡住**（§7.1）：只提醒，状态不变 |
| 待确认 needs_you | agent 提问、`questions.md` 有问题待答、被停止、运行中断（含后台等待被中断）、预算用完、**验收未通过**；纳入托管时任务还在进行中；未托管任务的 progress.md 写着"阻塞" |
| 待审核 in_review | 完成了，且设置了 `review: true` |
| 已完成 done | 新完成的任务带"未查看"标记，打开它的详情即标为已查看 |
| 已取消 cancelled | |
| 其他 unknown | 未托管任务的 progress.md 状态行不认识，或 state.json 无法解析 |

**"已完成"怎么定**：托管任务要同时满足两件事：agent 在 `progress.md` 里写了已完成，并且 `checks` 全部通过；没有配置检查的任务只看第一件。两个例外：
- **回答追问**：已完成的任务又跑一轮，如果主报告的路径和 mtime 与本轮开始前相比都没变，就算作回答追问。这时不跑检查，任务仍是已完成，也不再标为未查看。只对本轮开始前是"已完成"的任务这样处理。
- **纳入托管时已完成**：未托管任务的 progress.md 已经写了已完成，纳入时直接记为已完成（`review: true` 时为待审核），不跑检查。

**托管和未托管**：托管任务的状态在 `.agent_sys/state.json`，只由系统写。等待原因、运行中的子状态这类能从别的文件推出来的信息不存进去，扫描时现算。未托管任务的状态按 `progress.md` 开头的"状态:"行推导：

| progress.md | 状态 |
|---|---|
| 没有这个文件 | 草稿 |
| 进行中 / In progress / running | 运行中 |
| 阻塞 / Blocked | 待确认 |
| 已完成 / Done | 已完成；`review: true` 时为待审核，直到审核通过（frontmatter 写入 `approved_at`）。之后 progress.md 又报了更晚的完成时间（例如你在终端里接着做了），回到待审核 |
| 其他 | 其他 |

### 3.2 机器占用规则

- 托管任务在运行中和待确认时占用它声明的机器，直到进入待审核、已完成或已取消。已完成或待审核的任务再跑一轮（回答追问、按修改意见改）时重新占用。
- 待确认也算占用：等人回复期间，部署往往还在节点上。
- 机器按主机名识别，规则跨项目生效。
- 一个待开始的任务，只要它要的机器有一台被别的任务占着，调度器就不启动它；卡片写明"等待机器：node-a（被 experiment/e3 占用）"。
- 任何方式开始一轮运行（开始、发消息、继续、接管、追加预算）都会检查机器：被别的任务占用就拒绝，并说明占用者。任务自己已经占着这些机器时（例如在待确认时收到回复）也照样检查。要越过这条规则，只能改任务的机器列表。
- 正常情况下一台机器只有一个占用者。规则被绕过时（例如直接改文件，给一个待确认的任务加上别人占着的机器），所有占用者都会列出来，看板和首页一个占用者一项。
- 占着机器、又在待确认里等了 30 分钟以上的任务，首页和看板会标出"闲置占用"和时长。这直接针对"agent 睡着、节点空转一夜"。
- 已知空子：读不了的项目（project.yaml 解析失败等）不参与占用计算，它的任务占着的机器不会挡住别人。

## 4. 前端设计

技术：Preact + htm，不需要构建，ES module 直接加载；文件划分见 `web/README.md`。
- 和 API 同源，脚本只从本站加载（CSP 见 §5.9）。
- 只有一条 SSE 推送变化；运行输出收到通知后再去取增量。
- 中英双语：`agent_sys.yaml` 的 `language`，默认英文，界面右上角可切换。
- 默认暗色，可切换为跟随系统或亮色。

### 4.1 页面

| 页面 | 路由 | 一句话 |
|---|---|---|
| **首页** | `#/` | 所有项目里需要你的事、运行中的任务和机器 |
| **看板** | `#/p/<p>` | 一个项目的任务图、机器和待处理 |
| **台账** | `#/p/<p>/ledger` | 一个项目的结果、曲线和判定 |
| **任务抽屉** | 任意页面加 `?t=<p>/<类型>/<名字>` | 一个任务的全部：运行输出、工作区文件、配置 |
| **新建任务** | 任意页面加 `?new=<p>` | 新建任务的抽屉 |
| **规则 / 经验 / 设置** | `#/rules`、`#/knowledge`、`#/settings`；项目的在 `#/p/<p>/…` 下 | 注入的规则、经验库、全局和项目设置 |
| **已归档项目** | `#/archived` | 归档的项目：不显示在首页，任务不会自动开始，也不能新建任务或启动运行 |

#### 顶栏

所有页面共用：logo 和导航（在项目里时是看板、台账、规则、经验、设置）、待处理数（待确认 + 疑似卡住 + 待审核，不含已归档项目）、今日花费、与 server 的连接状态（断开时页面顶部有横幅）、语言切换、主题切换（暗色 → 跟随系统 → 亮色）。标签页标题上也显示待处理数。

#### 首页

从上到下：
1. 三张卡片：
   - **项目**：一行一个，带待确认、待审核、**待审核模板**等标记，每行有新建任务和归档按钮；"新建项目"在卡片标题旁。项目超过 8 个时出现筛选框。
   - **新建任务**：选项目后在首页打开新建任务抽屉（快捷键 n）。
   - **运行中**：运行中的任务，附各项目今日花费。
2. **机器**：所有项目被占用的机器，以及闲置占用告警；占用者可以点开。
3. **待确认**：一行一件，按原因给主按钮：回复 / 回答问题 / 查看检查 / 追加预算 / 继续 / 纳入托管。疑似卡住的运行中任务排在最前，按钮是发消息、停止、在终端中接管。两个以上任务运行中断时有"继续全部中断任务"。
4. **待审核**（主按钮"查看报告并审核"）和**最近完成**（可全部标为已查看）。

空的队列折叠成一行。

#### 看板

- **头部**：标题、**目标**、**基线和当前最佳（Δ%）**、自动开始开关、名额、新建任务。
- **待审核模板**（模板预置的任务还没开始时）：
  - 按执行顺序列出这些任务，以及项目规则和设置的入口；
  - **开始前检查**的结果（§5.5），每条带任务链接和改法；没有问题时显示一行"没有发现问题"；"重新检查"按钮；
  - "开始运行"，有错误时不可用。
- **机器条**：每台机器被哪个任务占着（每个占用者一项，可点开）、占了多久、占用者的状态；闲置占用标出来。
- **待处理摘要**：待确认、待审核的任务，以及**记录提醒**（依赖写错、成环、prompt 解析不了等）。
- **工具栏**：搜索（快捷键 /）、按状态和泳道筛选、按日期筛选（全部 / 3 天 / 7 天 / 30 天 / 自定义），以及"视图"菜单：折叠已完成、省略可由其他依赖推出的线、缩放和适应宽度、按种类显示依赖线（评审覆盖默认不画）。任务超过 40 个时默认折叠已完成，超过 60 个时默认只看最近 7 天。
- **依赖图**：泳道 × 日期。
  - 布局：按执行顺序从左上到右下，上游总在下游之上；同时创建的任务按依赖深度和泳道排。草稿放在画布底部单独一带。
  - 依赖线：直角折线，只走泳道之间和行之间的空隙；同一空隙里的线各占一条轨道，并按少交叉排序。可由其他依赖推出的产出、代码基线依赖默认不画（"视图"里或图上方可切换为全部显示），`children` 依赖总是画出。`children` 依赖画成青色点划线，提示里写子任务的完成数。
  - 选中一张卡片时，高亮它的上下游链。
  - 拖泳道标题可以调整泳道顺序（键盘：Alt+← / Alt+→）；把卡片拖到别的泳道就是改它的类型。
  - 卡片上有状态、机器、检查 ✓/✗、结果 Δ%。已完成或待审核的任务，上游在它开始后变了时有"上游已更新"标记（悬停看变了什么）。
- **任务列表**：在图下方，可以按列排序，有花费列。

#### 台账

- **头部**：主指标及方向、基线值、最佳值和 Δ%、结果数（作废、格式错误）。不按工作点比较时，`ledger.baseline` 指定的任务或文件没有可用结果会给出提醒。
- **曲线**（项目配置了 X 指标时）：每个系列一条 Y–X 折线（沿前沿），标出工作点和各系列在工作点的值与 Δ，附系列表；每个系列可以单独显示或隐藏。
- **趋势图**：每个结果一个点，形状和颜色表示判定（§4.3）。不按工作点比较时，还画基线虚线和"目前最佳"阶梯线。可按判定筛选。
- **结果表**：台账 Δ 和判定 Δ 分两列。行上可能有这些标记：
  - "判定未核实"及原因（复算对不上或无法复算，§5.6）；
  - "相对 v0 判定"（判定用的基线系列不是当前的）；
  - "复测"（同系列里非封版任务测的点）；
  - "筛选"（不是正式测量），附原因；
  - 噪声底的固定部分高于接受线时，噪声判定显示为"判不了 · 待集成"，不显示成普通的"噪声"。
- **格式错误的文件**列在最后。
- 点一行或一个点，打开任务抽屉并定位到这个结果文件。

各项的计算规则见 §5.6。

#### 任务抽屉

- **头部**：状态和原因、标题、花费/预算、模型、进度（已勾选/全部）、**机器**、**检查摘要**（✓/✗，点开是验收检查视图）、记录提醒、"在看板中定位"。
  - **上游已更新**时，列出每个上游变了什么，附"发消息复核"：预填好让任务按新结果复核的消息，可以改后再发。
  - 这个任务起草了草稿时："这个任务起草了 N 个任务：…"，附"全部纳入托管"（子任务和直接依赖它的草稿一起纳入，§6.1）。
- **按钮**按状态出现：开始、继续、停止、在终端中接管 / 结束接管、追加预算、审核通过、纳入托管；"更多"里是取消任务和删除任务。
- **三个 tab**（URL 参数 `tab=run|docs|task`，默认按状态选；←/→ 切换）：
  - **运行输出**：实时输出，检查的过程和结果也在流里；可以切换查看以前的轮次，查看本轮注入内容（launch.md）。终端接管时这里显示接管面板。底部是输入框：
    - 运行中发的消息：会话正在执行工具时，当前工具结束后读到；会话在两轮之间时，下一轮读到；会话在等后台命令或正在跑验收检查时，立即送达。
    - 待确认、待审核、已完成时，发消息就开始新一轮；待开始、草稿和已取消的任务不能发消息。
    - 运行中断的任务，输入框为空时按钮是"继续"。
  - **工作区文件**（`file=<路径>` 定位到文件）：报告和它的审核条（审核通过 / 发送修改意见）、questions.md 表单、交接清单、**验收检查**视图（`file=:checks`，每次尝试的每条检查）、结果文件；任意文件都能查看，Markdown 文件可以编辑。
  - **任务配置**：prompt、上游任务、类型和名字、机器、检查、agent 配置。可以只保存，也可以"保存并通知 agent"（把新 prompt 告诉已运行过的任务）。
- 打开一个已完成的任务，它就标为已查看。

#### 新建任务

- 选类型：prompt 按类型骨架预填，检查也一起预填；也可以在这里新建类型。
- 写标题和 prompt：输入 `@` 列出项目的任务，选中后写成 `@<类型>/<名字>`。
- 选上游任务，每个上游选依赖种类；边写边预览 prompt 和所选上游给出的依赖，以及名字能不能用。
- 选机器，设审核、预算、模型、effort、ultracode 和额外的 claude 参数。
- 可以直接排队，也可以存为草稿。

#### 规则、经验和设置

- **规则页**：每次注入的各段及大小（包括 agent_sys 约定和 **skills 索引**），编辑和新建全局 / 项目规则。
- **经验页**：处理候选（采纳、改写为规则、丢弃），编辑经验库和 INDEX.md。
- **设置页**：
  - 全局：界面语言、agent_sys 约定的语言（不决定 agent 的工作语言）、agent 的 Claude Code 配置（独立 / 继承，§5.9）、全局名额、疑似卡住提醒的分钟数（§7.1）；另列出各项目的默认设置。
  - 项目：标题、目标、机器、台账主指标和方向、自动开始、名额、新建任务的默认 agent 参数；归档 / 取消归档。台账的其他口径（X、工作点、噪声带、接受线……）在 project.yaml 里改。

#### 快捷键

| 键 | 作用 |
|---|---|
| n | 新建任务（在项目页里，或首页选好项目后） |
| g | 回首页 |
| / | 看板：聚焦搜索框 |
| Alt+← / Alt+→ | 看板：移动聚焦的泳道 |
| ← / → | 任务抽屉：切换 tab |
| Ctrl/⌘+Enter | 输入框：发送 |
| Esc | 关闭抽屉、弹出菜单 |

### 4.2 交互要点

- **agent 起草的子任务**：显示为草稿卡片。起草它们的任务的抽屉里列出"这个任务起草了 N 个任务"，点"全部纳入托管"一次批准一组；也可以逐个修改后再批准。纳入前先对这组草稿做开始前检查（§5.5）：有错误时一个也不纳入，对话框列出全部问题；只有提醒时列出后让你确认。
- **回答问题**：agent 有多个问题时，按约定写 questions.md 表格（`# | 问题 | 建议 | 你的意见`，单元格里的 `|` 写成 `\|`）。
  - 卡片主按钮"回答 N 个问题"打开表单。你的回答写回表格的"你的意见"一列（没填的行按建议处理，其余内容原样保留），并告诉同一会话接着做。
  - 任务此时开不了跑（prompt 有错误、预算用完、机器被占用）时拒绝提交，什么也不写。
  - 本轮写的 questions.md 里没有"你的意见"表时，主按钮是"回答问题"：打开这个文件，直接回复即可。
- **验收未通过**：卡片主按钮是"查看检查"，打开工作区文件 tab 的验收检查视图，列出每次尝试的每条检查和输出。回复"继续"后，agent 在同一会话里接着修。
- **机器冲突**：卡片上的等待原因写明"等待机器：X（被 Y 占用）"；要找占用者，从看板机器条或首页机器卡片点进去。占用者在待确认里等了 30 分钟以上时，那一项带"闲置占用"标签（§3.2）。
- **台账和任务**：卡片上的结果徽章显示最好的 Δ；台账的行和点打开任务抽屉，定位到对应的结果文件。
- **继续全部中断任务**：机器重启之类的原因让多个任务同时运行中断时，首页一个按钮让它们全部接着跑。
- 不做：燃尽图、故事点、多用户权限。

### 4.3 视觉

- 页面底色带一点冷绿。
- 状态色：待确认橙、运行中蓝、待审核紫、完成绿。卡片左侧色条标状态；未托管任务的色条是点线。
- logo：圆角方块里一条带节点、总体向上的折线（一串任务把指标推高），也用作 favicon。
- 动效尽量少：持续转动的只有运行中图标（一段弧绕着中心点转）；首屏加载时 logo 的折线自己画出来，页面内加载是一条扫过的细光带；提示消息滑入，开关和卡片有短暂的过渡。遵守 `prefers-reduced-motion`。
- 台账的判定：接受用完成绿，拒绝用警告红的叉，噪声用灰，"判不了 · 待集成"是灰色虚线空心圈，需重跑用待确认橙，未判定是中性的实心点，作废是空心点，基线是菱形；曲线上筛选的点是虚线方块，每个系列一个分类色。

## 5. 后端设计

### 5.1 进程

| 进程 | 职责 | 生命周期 |
|---|---|---|
| `agent-sys-v2 serve` | 扫描、调度、跟踪运行、REST + SSE、提供前端 | 放在 `tmux -L $AGENT_SYS_TMUX`（默认 `agent_sys`）的 `server` 窗口里（`--foreground` 则在当前终端）。可随时重启，正在跑的任务不受影响。一个 `AGENT_SYS_HOME` 只能有一个 server（home 里的锁文件），否则两个调度器会把同一个任务启动两次 |
| runner（`python -m agent_sys_v2.runner <run_dir>`） | 一轮会话：启动或恢复 claude 会话，转发消息和停止指令，记录事件；一轮结束时如果 agent 报告完成，执行检查 | 每轮一个，各自在一个 tmux 窗口里。会话空闲、没有后台命令、收件箱已读完（检查也跑完）时退出 |
| 终端接管（`runner <run_dir> --terminal`） | 包装进程，在 tmux 窗口里对同一会话开交互式 claude；claude 退出后，progress.md 已完成就执行一次检查，再写结束记录 | 你退出 claude 或在界面上点"结束接管"后结束；server 按 progress.md 和检查结果接回 |
| 手动回放（`runner <run_dir> --replay <events.jsonl> [--speed N] [--max-gap S]`） | 把录好的会话按原节奏写进 run 目录（目录不存在就新建），不调用模型；用于测试和界面开发。配置了 `agent.replay` 的任务（§5.2）由 server 照常启动 runner，launch.json 的 `kind` 是 `replay` | 回放完退出 |

server 和 runner 只通过 run 目录里的文件交换信息：

| 文件 | 写入方 | 内容 |
|---|---|---|
| `launch.json` | server | 这一轮的全部启动参数，见下 |
| `system.md` | server | 注入内容（§5.7） |
| `launch.md` | server | 给人看的注入清单、Claude Code 配置（§5.9）、验收依据、首条消息和等价命令 |
| `inbox.jsonl` | server 追加 | 你的消息、停止指令 |
| `events.jsonl` | runner | CLI 的原始帧，加上 runner 自己的事件（`thinking_tokens`、`ui_status` 等界面用不到的帧不落盘） |
| `run.json` | runner | 开始记录和结束记录（退出原因、会话累计花费） |
| `checks.json` | runner | 每一次检查尝试的结果 |
| `stderr.log` | runner | runner 自己和 claude CLI 的错误输出（SDK 运行时） |

`launch.json` 的字段：
- 运行本身：`id`、`kind`（`sdk` / `terminal` / `replay`）、`task`、`project`、`cwd`、`session_id`、`resume`、`window`、`started`、`message`（首条消息）、`system_known`（agent 是否会读到本轮的 system.md）。
- agent 配置：`model`、`effort`、`ultracode`、`max_budget_usd`（本轮还剩的预算）、`cost_start_usd`、`extra_args`、`cli_path`、`replay`、`language`（界面语言）、`agent_language`（`protocol_language`）。
- 环境：`env`（§5.10）、`claude`（Claude Code 配置：配置目录、转发和没转发的键，§5.9）。
- 验收：`checks`、`check_retries`、`ledger`（台账口径快照）、`toolchain`（检查用到的文件的摘要），见 §5.4。
- 对照：`before`（本轮开始前任务的状态、主报告和它的 mtime，用于判断"回答追问"，§3.1）、`upstream`（阻塞上游当时的摘要，§5.5）。
- 其他：`system_md`（system.md 的路径）。

### 5.2 文件布局与配置

```
$AGENT_SYS_HOME/                       默认 ~/agent_sys_home
  agent_sys.yaml                       全局配置，见下
  claude/                              agent 自己的 Claude Code 配置目录（CLAUDE_CONFIG_DIR，§5.9）：
                                       settings.json、.claude.json 每次启动时生成或补齐，projects/ 链接到 ~/.claude/projects
  rules/*.md                           全局规则
  knowledge/{INDEX.md,entries/}        全局经验库
  projects/<p>/
    project.yaml                       项目配置，见下
    rules/*.md  knowledge/  skills/    项目的规则、经验、自带 skills
    task_types/<类型>.md                prompt 骨架（frontmatter 可带默认的 review、checks）
    tasks/<类型>/<任务>.md               任务 prompt（frontmatter 见下）
    workspaces/<类型>/<任务>/
      progress.md report.md issue.md questions.md lessons/ log/ results/
      .agent_sys/state.json            托管状态（只由系统写）
      .agent_sys/runs/<run>/           一轮的完整记录（§5.1）
    repos/                             代码仓库，不扫描
    .trash/                            删除的任务（prompt 和工作区一起移进来）
```

**主报告**：工作区里的 `report.md`；没有时取文件名含 report 的其他 `.md`，再没有时取 `reports/` 下编号最新的一份。回答追问的判断（§3.1）和上游已更新（§5.5）都用它。

**`agent_sys.yaml`**（全局，设置页可改）：

| 键 | 默认 | 含义 |
|---|---|---|
| `language` | `en` | 界面和 agent_sys 提示信息的语言（`zh` / `en`） |
| `protocol_language` | `zh` | 注入给 agent 的约定、system.md 的开头和 agent_sys 发给 agent 的消息用的语言。agent 的工作语言跟任务 prompt 走，不受它影响 |
| `max_running` | 4 | 全局名额：所有项目同时运行的任务数上限 |
| `claude_config` | `isolated` | agent 的 Claude Code 配置：`isolated` 用专用目录，`inherit` 用你的 `~/.claude`（§5.9） |
| `stall_minutes` | 15 | 疑似卡住提醒的分钟数，可以是小数，0 关闭（§7.1） |
| `cli_path` | 随 SDK 自带的 claude | 换一个 claude CLI |

**`project.yaml`**（`infera-perf` 模板填好之后的样子）：

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
  primary: input_tput_per_gpu   # 主指标（曲线的 Y）
  higher_is_better: true
  x: p90_intvty                 # 曲线的 X（可选；x_higher_is_better 默认 true）
  baseline_series: v0           # 基线系列：封版后由 close 改成 vN
  working_point: {x: 30}        # 判 A/B 的工作点：{x: X} 或 {conc: C}
  noise_pct: 2                  # A/B 的噪声带（%）
  accept_pct: 3                 # A/B 的接受线（%）
  min_duration_s: 3600          # 正式测量的最短设定时长
  reproduce_pct: 5              # 基线臂复现基线系列的容差（%）
  max_arm_gap_min: 60           # 两次测量之间的最大间隔（分钟）
  min_repeats: 2                # baseline 在工作点同节点的正式测量次数
quality: {gsm8k_margin: 0.02}   # 封版时 GSM8K 的非劣界
skills: [infera-deploy, infera-bench, infera-quality-gate, infera-profile, infera-diagnose, infera-experiment, infera-hygiene]
autorun: true
max_running: 3
lanes: [intake, survey, baseline, profile, plan, experiment, close, develop, review]
agent: {model: opus, effort: high, ultracode: false, budget_usd: 200, extra_args: []}
```

- `ledger` 和 `quality` 的含义、默认值和谁能改见 §5.6。台账口径只由你改，命令行只能更严；每轮运行开始时做一次快照（§5.4）。
- `ledger.baseline`（可选）把基线钉到一个任务或一个结果文件（§5.6）。
- `autorun`（默认 true）：自动开始；`max_running`（默认 3）：项目名额；`lanes`：泳道顺序。
- `agent`：新建任务的默认值，任务 frontmatter 的 `agent` 逐键覆盖。
  - `model`、`effort`、`budget_usd`（默认 opus、high、200 美元）。
  - `ultracode`（默认 false）：为 true 时给 claude 的 `--settings` 加 `"ultracode": true`。
  - `extra_args`：额外的 claude 命令行参数，例如项目需要的 `--mcp-config`、`--plugin-dir`（§5.9）；解析不了时拒绝开始。
  - `replay`：不调用模型，回放录好的会话（演示和测试用）。可以写一个 events.jsonl 路径（相对项目目录）、一个列表（第 n 轮回放第 n 个，之后一直用最后一个），或 `{events, speed, max_gap, writes}`；`writes` 是回放结束时写进工作区的文件（`{相对路径: 内容}`，模拟 agent 写 progress.md、report.md、questions.md），这样不调用模型也能演完一串任务、提问和审核。回放的运行在终端接管时开的是 bash，不是 claude。
- 系统写的键：模板预置的任务还没审核时，有一行 `template_review: true`（§6.1）；归档时写 `archived_at`。

**任务 frontmatter**：

| 键 | 含义 |
|---|---|
| `title`、`summary` | 标题（默认用名字）、一句话说明 |
| `depends_on`、`parent` | 依赖和父任务（§3） |
| `review` | `true` 时完成后进待审核 |
| `machines` | 要用的机器（§3.2） |
| `checks`、`check_retries` | 验收检查和重试次数（§5.4） |
| `agent` | 覆盖项目的 `agent` 默认值 |
| `approved_at` | 系统写：未托管任务审核通过的时间（§3.1） |

模板还会用到自己的键，例如 infera-perf 实验子任务的 `expect`（§6.1）。

### 5.3 模块

一个 Python 包 `agent_sys_v2`（目录 `Infera/agent_sys_v2/`，src 布局，与 v1 的 `agent_sys/` 并列）。

| 模块 | 职责 |
|---|---|
| `model.py` | 数据类型（Project、Task、Edge、TaskType）、frontmatter 读写、state.json 和 run.json 的格式、未托管任务的状态推导、`agent_sys.yaml` 和 project.yaml 的默认值、文件读写工具 |
| `scan.py` | 扫描一个项目：任务、依赖边、等待原因、记录提醒（依赖、parent、环的判断和开始前检查共用）；Watcher 按 stat 发现变化 |
| `preflight.py` | 开始前整体检查：一次列出项目里会挡住运行的问题（§5.5） |
| `upstream.py` | 上游已更新：开始一轮时记下阻塞上游的摘要，扫描时比较，给出提示和复核消息（§5.5） |
| `ops.py` | 写操作：新建项目（按模板）、类型、任务，改名，删除，审核，纳入托管和批量纳入子任务草稿，归档，开始运行，泳道顺序，设置 |
| `sched.py` | 调度：启动待开始的任务（依赖、名额、机器） |
| `machines.py` | 解析机器列表；占用表；闲置占用；互斥检查 |
| `runs.py` | 开始、消息、停止、接管、预算、取消、一轮结束后的状态迁移；Monitor（server 跟踪运行中任务的循环） |
| `runner.py` | 一轮会话的进程；检查的执行和反馈；终端接管的包装进程；回放 |
| `events.py` | 读 events.jsonl：给运行输出的精简事件（分页、单条全文），以及跟踪运行中会话的状态和疑似卡住的判定（§7.1） |
| `procs.py` | 在 /proc 里找一轮运行的进程（runner、claude 会话、检查）并结束它们 |
| `replay.py` | 回放录好的 events.jsonl，不调用模型（测试、演示、界面开发） |
| `launch.py` | system.md（约定、规则、skills 索引、经验索引）、首条消息和后续消息、SDK 参数、launch.json 和 launch.md |
| `claude_config.py` | agent 的 Claude Code 配置：专用目录、env 和 settings 键的白名单、`projects/` 软链、首次启动状态、退路 `claude_config: inherit`（§5.9） |
| `tmux.py` | agent_sys 的 tmux server 和窗口；交给子进程的干净环境 |
| `checks.py` | 解析 checks、执行（超时、输出截断）、记录和汇总、失败反馈、检查用到的文件的摘要 |
| `ledger.py` | 读结果文件、校验格式、算 Δ（参照函数 `reference`）、最佳值和曲线；命令行 `ledger check`、`verify`、`agentx`、`gsm8k`（§5.6） |
| `verdict.py` | A/B 判定：判定函数和带门的判定入口 `judge_ab`（ab_judge 和复算共用）、`verdict.inputs`、两类作废、口径和快照、从 AgentX 原始输出复算（`ledger verify`、台账的"判定未核实"） |
| `gates.py` | 一个 AgentX 点的测量信息（AIPerf 的窗口、节点、设定时长、fast 模式），正式和筛选，A/B 的门：基线复现、测量时序、复测噪声，以及门怎么改判定 |
| `gsm8k.py` | GSM8K：lm_eval 逐题记录转成精度结果、按逐题记录复算、Wilson / Newcombe 区间（拷自 v1 的 `eval_stats.py`）、accept 的门和封版的门 |
| `perfcheck.py` | `infera-perf` 模板一轮里各任务的字段检查，供验收检查调用（§6.1） |
| `skills.py` | 选出项目可用的 skills，生成 skills 索引 |
| `questions.py` | questions.md 表单：读表格、写回答、判断是不是本轮的问题 |
| `knowledge.py` | 规则页和经验页：规则读写、经验候选的采纳和丢弃、经验库和 INDEX.md |
| `files.py` | 工作区文件的列目录、读、写 |
| `i18n.py` | 中文源文字的英文翻译（界面和给 agent 的文字） |
| `server.py` | REST + SSE，提供前端；Host/Origin 校验 |
| `cli.py` | `agent-sys-v2` 命令（见下）；一个 home 一个 server 的锁 |

另有 `web/`（前端，见 `web/README.md`）、`templates/`（注入的约定 `_common/rules/protocol.md` 和 `protocol.en.md`，项目模板 `blank`、`infera-perf`）、`skills/`（内置 skills）、`tools/`（`demo.py` 生成演示 home；`stall_calibrate.py` 标定疑似卡住的阈值，§7.1）。

**命令行**（所有子命令都接受 `--home DIR`）：

| 命令 | 作用 |
|---|---|
| `agent-sys-v2 serve [--sock PATH] [--foreground]` | 启动 server；socket 默认 `/tmp/agent_sys-<uid>/agent_sys.sock` |
| `agent-sys-v2 new <项目> [--template blank\|infera-perf] [--title T]` | 用模板新建项目（默认 `blank`） |
| `agent-sys-v2 check <项目名或目录> [--drafts-of TASK]` | 开始前整体检查（§5.5），有错误时退出码 1 |
| `agent-sys-v2 --version` | 版本 |

### 5.4 一轮运行和验收检查

```
start_run → 写 run 目录 → tmux 开 runner
runner：connect / resume → 发首条消息或你的消息 → 转发收件箱 → 记录帧
  一轮结束（会话空闲、没有后台命令、收件箱已读完）：
    progress.md 没写"已完成"              → 退出（server 判为待确认：提问 / 有问题待答）
    写了"已完成"，只是回答了追问          → 退出，不跑检查（仍为已完成）
    写了"已完成"，且有检查                 → 逐条执行检查（runner 事件 checks / check，带 phase，流进运行输出）
        全部通过                           → 退出（server 判为已完成或待审核）
        有失败，还有重试次数               → 把失败的检查和输出尾部作为一条消息发回同一会话，等它再次标记完成
        有失败，重试次数用完               → 退出（server 判为待确认 · 验收未通过）
```

**执行**
- 每条检查用 `bash -c` 执行，cwd 是项目目录，按顺序全部执行（一条失败不跳过后面的）。
- 环境变量：`AGENT_SYS_PYTHON`（运行 agent_sys 的 Python，用它调用 `agent_sys_v2.ledger` 等工具，不依赖 PATH）、`AGENT_SYS_PROJECT`、`AGENT_SYS_TASK`、`AGENT_SYS_WORKSPACE`、`AGENT_SYS_MACHINES`（逗号分隔）、`AGENT_SYS_SKILLS`（内置 skills 的目录）。agent 自己的环境里也有这几个变量。检查另有 `AGENT_SYS_CHECK_RUN`、`AGENT_SYS_RULES` 和 `PYTHONSAFEPATH=1`（见下文"口径快照"）。
- 默认超时 600 s，超时就杀掉整个进程组；输出只留最后 4000 字符。

**事件和记录**：runner 把 `checks`（phase start / end）和每一条 `check`（phase start / end）写进 events.jsonl，运行输出实时显示；每一次尝试追加到 `checks.json`。server 读最后一次尝试，决定是已完成还是验收未通过；任务抽屉和卡片都读它。

**反馈**：检查列在给 agent 的首条消息里。失败时发回 agent 的消息写明哪几条检查失败、退出码或超时、输出尾部，并要求修好后再标记完成；用 `protocol_language` 写。重试次数是 frontmatter 的 `check_retries`，默认 2，即最多执行 3 次；写的不是非负整数时按 2 算。

**检查期间**：runner 照常读收件箱，你发的消息立即送进会话，这一轮接着跑。**停止**会杀掉正在执行的检查和它启动的进程（检查进程带 `AGENT_SYS_CHECK_RUN` 标记）；这次尝试不记录，运行输出里显示为"随停止中断"而不是失败；会话此时空闲，不再发 interrupt。runner 意外退出时，server 结束它留下的检查进程。

**追问不重跑**：见 §3.1。

**解析失败就拒绝运行**：任务的 frontmatter 或 checks 解析不了时（扫描时发现），任何方式开始运行都会被拒绝，否则会不带检查、用默认的机器和预算跑起来。错误写在任务的提醒里；待开始的任务等待原因是"prompt 有错误"，调度器不启动它，改正后才会开始。

**终端接管**：
- claude 退出后，包装进程在 progress.md 已完成时执行一次检查（没有会话可以发回，所以不重试），结果打印在终端里并写进 `checks.json`，然后写结束记录。
- 界面上的"结束接管"结束交互式 claude，最多等包装进程 5 s。检查还在跑就让它跑完（界面显示正在跑的检查，不再提供"结束接管"），结束记录写好后 Monitor 再推进任务。
- 终端接管不受预算限制：开始时不检查预算，交互式 claude 也不带 `--max-budget-usd`（这个参数只在 `--print` 模式下有效）。

**检查用到的文件防改**：
- 开始一轮时，launch.json 的 `toolchain` 记下检查工具链的 sha256，包括：本包里检查会调用的模块（`ledger`、`perfcheck` 和检查命令里写出的 `agent_sys_v2.<模块>`，按 import 关系展开）；内置 skills 和项目 skills 的 `skills/<名字>/scripts/` 下的全部文件；检查命令里直接写出路径、不在 `workspaces/` 下的文件；项目目录下名为 `agent_sys_v2*` 的文件和目录（目录则算其中全部文件）。
- 每次执行检查前重新计算并比对。有变化就在这次尝试的最前面记一条必定失败的检查（"the files the checks use are unchanged"），列出改了、加了、删了哪些文件；agent 收到的反馈里同样有这份清单。
- 目的是防止 agent 顺手把检查改松，不防恶意。
- agent_sys_v2 和项目要改的 Infera 仓库共用一个 checkout 时，在那里 `git checkout` / `pull` 也会触发它，所以报错里提示从单独的 checkout 或 wheel 安装。
- launch.json 里没有 `toolchain`（旧版本写的）时不比对。

**口径快照**：
- 开始一轮时，把项目的台账口径快照进 launch.json 的 `ledger`：`ledger.primary`、`higher_is_better`、`x`、`x_higher_is_better`、`baseline_series`、`noise_pct`、`accept_pct`、`min_duration_s`、`reproduce_pct`、`max_arm_gap_min`、`min_repeats`，以及 `quality.gsm8k_margin`。launch.md 的"验收依据"一节列出快照和检查文件的个数。
- launch.json 在 agent 能写的工作区里，所以 runner 把它启动时（agent 开始之前）读到的快照经环境变量 `AGENT_SYS_RULES` 交给检查。
- 检查还带 `PYTHONSAFEPATH=1`，防止项目目录下同名的 `agent_sys_v2` 被优先 import；Python 3.10 不认这个变量，所以项目目录下的 `agent_sys_v2*` 也算进检查文件，被改了能发现。
- `ledger verify` 和 `perfcheck` 按快照判（§5.6、§6.1）。

**默认检查**：类型骨架 `task_types/<类型>.md` 的 frontmatter 可以带默认的 `checks`（和 `review`）。在界面上新建任务时预填；agent 起草的草稿纳入托管时，frontmatter 没写的就取类型的默认值。

写法：

```yaml
checks:
  - test -s "$AGENT_SYS_WORKSPACE/report.md"
  - {name: results, run: '"$AGENT_SYS_PYTHON" -m agent_sys_v2.ledger verify "$AGENT_SYS_WORKSPACE/results"', timeout: 60}
check_retries: 2
```

### 5.5 调度

调度每 2 s 执行一次，在扫描之后，只在 server 里跑：没有 server 就没有调度。

**等待原因**：待开始的任务按下面的顺序，取第一个成立的原因：

| # | 原因 | 范围 |
|---|---|---|
| 1 | 项目已归档 | 本项目，扫描时算出 |
| 2 | prompt 有错误（frontmatter 或 checks 解析不了，改正后才会开始；§5.4） | 本项目，扫描时算出 |
| 3 | 阻塞类上游未完成 | 本项目，扫描时算出 |
| 4 | 等上游的子任务（`children` 依赖，§6.1） | 本项目，扫描时算出 |
| 5 | 等你审核模板（§6.1） | 本项目，扫描时算出 |
| 6 | 自动开始已关闭 | 本项目，扫描时算出 |
| 7 | **等待机器**：机器被别的任务占用（写明占用者） | 跨项目，调度器算出 |
| 8 | 项目名额已满（`max_running`，默认 3） | 跨项目，调度器算出 |
| 9 | 全局名额已满（`agent_sys.yaml` 的 `max_running`，默认 4） | 跨项目，调度器算出 |

没有等待原因的任务按等待时间从长到短启动，一次启动一个，每启动一个都重新扫描。运行中的任务占一个名额，在后台等待和终端接管期间也算；未托管的任务不占。

**开始的规则**：
- **机器互斥在 `start_run` 里检查**，不只在调度器里：手动开始、发消息、继续、接管、追加预算都经过 `start_run`，在写锁下读所有项目的最新状态，机器被别的任务占用就拒绝；任务自己已占着机器时也检查（§3.2）。
- **手动开始**不看上游和名额，但仍然拒绝：项目已归档、预算用完（终端接管除外，§5.4）、frontmatter 或 checks 解析不了、`agent.extra_args` 解析不了、机器被占用。项目的模板还在审核中时，手动开始其中一个任务也就结束了审核（先做开始前检查，见下）。
- **自动开始失败**的任务进入待确认：预算不够时是"预算用完"，否则是"运行中断"并写明原因；不会自动重试。
- **runner 退出时才到的消息**：runner 决定退出的那一刻才送到、它没读到的消息，由 server 用它们自动开始下一轮。开始不了时（例如预算用完），任务留在待确认，这些消息存进 state.json 的 `unsent`，随下一轮一起发出。

#### 开始前整体检查

`preflight.preflight(project, tasks=None)` 返回 `[{task, level, message, fix}]`。它把项目文件里静态就能发现的问题一次列全，免得第 4 个任务跑起来才发现写错。
- 问题收集齐再返回，错误排在提醒前面。`task` 为空表示 project.yaml 本身的问题。
- frontmatter 或 checks 解析不了的任务只报这一条，不再往下查；草稿要带上的类型默认 checks 解析不了时也一样。
- `message` 写问题和冲突的两边（例如"有：…"列出已有的机器或 skills），`fix` 写怎么改。名字写错时给出最接近的已有名字（difflib；任务名按类型和名字分别比）。

| 级别 | 检查项 |
|---|---|
| error | 依赖：depends_on 看不懂、任务不存在、kind 不认识；parent 看不懂、不存在；阻塞类依赖成环。环上的每个任务报一个经过它的最短环（写出环上的任务），相同的环只报一次。和卡片上的记录提醒是同一套判断 |
| error | frontmatter 或 checks 解析不了；草稿没写 checks 时，纳入后会带上的类型默认 checks 解析不了 |
| error | 任务的 machines 不在 project.yaml 的 machines 里 |
| error | project.yaml 的 skills 里有内置和项目都没有的名字 |
| error | checks 调用的东西不存在：`-m agent_sys_v2.<模块>` 的模块；`perfcheck <类型>`、`ledger <子命令>` 不认识（参数是 `$变量` 或以 `-` 开头时不查）。草稿没写 checks 时查类型的默认 checks。命令按 shell 的规则拆成简单命令：`;`、`&&`、`\|` 等贴着词写也拆开；注释、重定向的目标、开头的 `if`/`!` 之类不算 |
| error | checks 里的路径不存在：以 `$AGENT_SYS_SKILLS/` 开头的；以 `$AGENT_SYS_PROJECT/` 开头、作为脚本执行（命令本身，或 python / bash 的第一个参数）的 |
| warn | checks 里 `$AGENT_SYS_PROJECT/` 下的其他路径不存在（可能由上游任务生成）。不查：`workspaces/` 下的路径、`test`/`[`/`[[` 的操作数、含 `$` 或通配符的路径、看不了的路径（名字太长、目录没有读权限） |
| warn | 检查命令无法静态分析（例如引号没配对） |
| warn | 任务的类型不在 project.yaml 的 lanes 里（配置了 lanes 时；每个类型报一次） |
| warn | 依赖了已取消的任务（阻塞类的会写明会一直等） |

不查 `rules/project.md` 的占位符和空的 machines：infera-perf 模板审核时，它们是留给 intake 填的。

有三个入口会做这项检查，都由服务端强制，界面只展示；有 error 就不开始，没有"仍然开始"：
- **开始运行**（`ops.start_project`），以及模板审核期间**手动开始**其中一个任务（它同样结束审核）：查已托管、未完成的任务（草稿在纳入时再查）。有错误时返回 409，`problems` 带整张清单，模板审核状态不变；看板的待审核模板横幅列出同一份清单，并有"重新检查"：改了检查用到的脚本或 skill 后（这些文件不在 Watcher 的范围里）用它刷新。审核结束后，单个任务的开始不再做整体检查（解析不了的仍按 §5.4 拒绝）。
- **全部纳入托管**（`ops.join_children`）：只查这组草稿，加上 project.yaml 本身的问题；有错误时一个也不纳入。单独**纳入托管**一个草稿（`ops.join`）时也先查它，否则可以借此绕过检查。界面不按自己手里可能过时的清单拦，交给服务端判。
- **`agent-sys-v2 check <项目名或目录> [--drafts-of <任务>]`**：查所有未完成的任务（含草稿），逐条打印，有错误时退出码 1，不写任何文件。`--drafts-of` 只查这个任务起草的草稿，和对它点"全部纳入托管"查的一样。infera-round 的 skill 要求 plan、intake、close 起草完任务后先跑 `"$AGENT_SYS_PYTHON" -m agent_sys_v2.cli check "$AGENT_SYS_PROJECT" --drafts-of "$AGENT_SYS_TASK"`；别的任务的问题写进报告，不替它改。

#### 上游已更新

`upstream.py` 只提示，不让下游失效，也不改状态。依据是 v1 的结论：每次执行记下所用输入的版本，上游重跑后，已跑完的下游保留旧版本，不做失效传播。这里不搬 v1 的版本槽，只记摘要。

- **记录**：开始一轮时，launch.json 的 `upstream` 记下每个 `input`、`code` 上游的：状态；主报告的路径和 sha256；`results/` 顶层文件（不含 `.` 开头的）和报告交接清单里**文件**的 `[大小, mtime_ns]`。目录不记，因为代码基线的 checkout 会随下游在里面构建而变；报告和 `results/` 本身不重复记。
- **比较**：扫描时，对已托管、已完成或待审核的任务，拿它**最后一轮**记下的摘要和上游现在比。状态变了、报告内容变了（只碰了 mtime 不算）、文件增删改了，就在卡片和详情里提示"上游 X 在本任务开始后更新了：……"。
- **复核**：详情里附预填好的复核消息（用 agent 的语言）。发出后在同一会话中开始新一轮，新一轮记下新摘要，提示随之消失。
- **不提示的情况**：上游被删除或改名；旧版本的 launch.json 没有 `upstream`，或读不了、格式不对。
- **多快出现**：`report.md`、`results/*.json` 的变化在 2 s 内；Watcher 不看的文件（其他名字的主报告、交接清单里的其他文件、`results/` 里改写的非 JSON 文件）要等每 10 分钟一次的全量扫描。
- **开销**：报告的 sha256 按（路径、大小、mtime）缓存，launch.json 按（路径、大小、mtime_ns）缓存，同一次扫描里一个上游只算一次。上游已完成或待审核时，交接清单的路径沿用扫描时 `scan.handoff` 的结果，不再逐个 resolve（NFS 上 resolve 每一级目录都要一次 stat）。实测（NFS）：一个有 120 个已完成任务、209 条 input/code 依赖的项目，一次扫描约 560 ms，这一步单独约 30 ms（约 5%）。

### 5.6 台账

台账把项目里所有任务测到的结果汇成一张表，算出基线、最佳值和曲线；A/B 判定写在结果文件里，台账和验收检查都能从原始输出复算它。性能只以 AgentX 为准，精度只以 GSM8K 为准。

**术语**：

| 术语 | 含义 |
|---|---|
| 结果文件 | `results/` 顶层、`schema: infera.result/v1` 的 JSON，一个测量点一个文件 |
| 系列（series） | 一条曲线：`v0`、`v1` 是版本，`r2-spf` 是一个实验，`ix:<条目>` 是对标 |
| 基线系列 | `ledger.baseline_series`：当前版本，A/B 的基线臂属于它 |
| 封版任务 | 测出一个版本的任务：`v0` 是类型为 baseline 的任务，`vN` 是 `close/rN` |
| 参照 | 一个系列里由封版任务测的有效正式点；Δ、前沿、工作点都只用它 |
| 复测 | 同一系列里别的任务测的点（例如实验的基线臂）：画出来，但不进参照 |
| 正式测量 / 筛选 | 时长够、不是 fast 模式的 AgentX 测量是正式的；其余是筛选：进台账，但不能作判定输入 |
| 台账 Δ / 判定 Δ | 台账 Δ 是相对基线系列参照的百分比；判定 Δ 是 A/B 自己算的，相对本实验的基线臂 |
| gain | Δ 按方向换成"正数更好" |
| 噪声带 / 接受线 | `ledger.noise_pct`、`accept_pct`（默认 2%、3%），只由你改 |
| 噪声底 | 一次 A/B 判"噪声"的界：噪声带、本版本的复测噪声、A/B 自身离散三者中最大的 |
| 可计入最佳 | 能当"最佳"的结果：有效的、判为接受的候选，或没有判定的基线 |

#### 读取和校验

- **读哪些文件**：每个任务工作区 `results/` 顶层的 `*.json`（子目录放原始数据，不读）。`schema` 以 `infera.result/` 开头的才算结果，其他 JSON 忽略。
- **缓存**：文件按 mtime 和大小缓存；Watcher 也只看 `results/*.json`，子目录里的原始数据变化不触发重新扫描。项目每次重新扫描时算一次（给看板头部和卡片的结果徽章），打开台账页时再算一次。
- **校验**：必填 `schema`、`label`、`metrics`（至少一个数字；NaN 和 Infinity 不算数字）。`role`、`round`、`series`、`verdict` 的取值也要对；`workload` 里的 `conc`、`duration_s`、`benchmark_duration_s`、`fast`、`node`、`window`（开始不晚于结束）有值时要对。格式错误的文件不进表，列在台账底部，任务卡片上的结果徽章也会标出来。

#### 基线、Δ 和参照

- **基线**：`ledger.baseline` 指定一个任务或一个结果文件时，取其中最新的、有效且带主指标的结果；没有指定时，取最新的、有效的 `role: baseline` 结果。指定的任务或文件没有这样的结果时，退回到后一种取法，台账返回 `pin_missing`（写明指定的是哪个），台账页给出提醒。
- **Δ 和 gain**：Δ（`delta_pct`，界面上叫"台账 Δ"）是主指标相对基线的百分比；gain 是同一个数按 `higher_is_better` 换成"正数更好"，界面按 gain 着色。作废的、判为噪声的和筛选的结果没有 gain（null），不着色。
- **按并发比**：有基线系列时（`ledger.baseline_series`；没写时取基线结果所在的系列），带并发（`workload.conc`）的结果和基线系列在同一并发的**参照**比（多个取平均）；参照没有这个并发时不给 Δ。
- **参照和复测**（`ledger.reference(table, series)`，A/B 的门也用它）：一个系列的参照只取封版任务测的有效正式点。同一系列里别的任务测的点（按 skill，实验的基线臂记成基线系列的复测）是**复测**（行上 `repeat: true`）：照样画在曲线上（方块），但不进参照、不进前沿、不进工作点的值，所以参照不会随实验增多而漂移。不是版本名的系列，或者没有封版任务测过的系列，参照是它的全部有效点。
- **正式测量和筛选**：结果行的 `screening` 写明它为什么不是正式测量（空串表示正式）。原因有四种：AgentX fast 模式（`workload.fast`）；设定时长低于 `ledger.min_duration_s`（默认 3600 s）；实测时长不到 `ledger.min_duration_s` 的 95%（借用 AIPerf 自己 `metric_duration_coverage` 的 95% 标准）；有 AgentX 来源但读不到实测时长。筛选的点照常进台账、画在曲线上（虚线方块），界面标"筛选"并写原因；但不进参照、前沿和工作点，不算同伴，也不能作判定输入。判定时按原始输出（运行目录）判，不看结果文件里写的。没有 AgentX 来源、什么都读不到的旧结果不算筛选。

#### 曲线、工作点和最佳

- **曲线**：结果带 `round`（轮次）和 `series`；`role: reference` 是别人的数（对标）。配置了 `ledger.x`（曲线的 X；`x_higher_is_better` 默认 true）时，每个系列是一条 Y–X 曲线：按 X 排序，标出前沿（同系列里没有别的点在两个轴上都不比它差）。
- **工作点**：`ledger.working_point` 是判 A/B 的工作点，只用参照点，不用复测：
  - `{conc: C}`：取该并发的点（多个取平均）；
  - `{x: X}`：取 X 不低于该值时的最大 Y（在前沿上相邻两点之间插值）。
  各系列在工作点的值，以及相对基线系列的 Δ 和 gain，显示在台账页的系列表里。
- **可计入最佳**：有效，并且是判定为接受的候选，或者是没有判定的基线（`role: reference` 是别人的数；带判定的基线不是 A/B）。噪声、需重跑、拒绝和还没判定的候选都不能当最佳。一个系列里除筛选点以外的点全部可计入最佳，这个系列才可计入最佳。台账页、看板头部和卡片徽章都按这一条规则（`ledger.counts`）。
- **头部**（台账页和看板）：
  - 基线系列能到达工作点时，按工作点比较系列：基线是基线系列在工作点的值，最佳是可计入最佳的系列里在工作点 gain 最大的那个。
  - 否则，如果 Δ 是按并发对基线系列算的（不同并发的值不可比），最佳是可计入最佳的结果里相对同并发基线 gain 最大的那个（都不比基线好时就是基线），没有"目前最佳"（`best_so_far` 为空）。
  - 再否则，最佳是可计入最佳的结果里主指标最好的那个，趋势图另画"目前最佳"阶梯线。
- **卡片徽章**：结果数、除基线外可计入最佳的结果里最好的 Δ、最近一个结果的判定、格式错误的文件数。

#### 判定在台账上的显示

- **判定 Δ**：`verdict_delta_pct` 是判定里写的 `vs_baseline_pct`，相对这次实验自己的基线臂；和台账 Δ 分两列显示。
- **相对旧基线判定**：判定所用基线臂所属的系列（`verdict.inputs.baseline_series`）不是当前基线系列时（例如封版后 `baseline_series` 从 v0 变成 v1），行上的 `judged_against` 写出它，界面标"相对 v0 判定"。
- **判定可复算**：带判定（decision、inputs 或作废）的结果，不论 role，台账都用 `ledger verify` 的同一套检查复算（不比较此刻的 project.yaml）。结果放在 `verified`（无判定时为 null）和 `unverified`（原因）里，界面标"判定未核实"并显示原因。没有 `inputs` 的旧判定照常显示、标"未核实"，不算格式错误。
- **复算的缓存**：按复算读过的所有文件和目录的 mtime 和大小缓存：结果文件、输入、原始输出、各任务的 `results/`、任务文件和 `tasks/`、`workspaces/` 目录、运行的 launch.json、作废的证据文件、GSM8K 的逐题记录、project.yaml。

#### 转换和检查工具

**`ledger agentx`**：把 InferenceX AgentX 的输出转成结果文件。

```
"$AGENT_SYS_PYTHON" -m agent_sys_v2.ledger agentx <agentx_conc*.json 或目录> --out "$AGENT_SYS_WORKSPACE/results" \
    --series v0 --round 1 --role baseline [--label L] [--change "…"] [--time ISO]
```

- 指标名和算法与 InferenceX App 入库一致（交互性 = 1/p90 full-response ITL）。
- 文件名和 `label` 是 `<label 或 series>-c<并发>`，所以用作前缀的那个不能含 `/`。不覆盖已有的结果文件，原子写入。`--time` 默认取原始文件的 mtime。
- `config` 记下 AgentX 报告的部署信息：`hw`、`image`、`model`、`framework`、`precision`、`spec_decoding`、`disagg`、`topology`、`kv_offloading`、`router`。
- `evidence` 记下来源：
  - `agentx`（原始文件；在项目目录下时是相对项目目录的路径，否则是绝对路径）和 `agentx_sha256`；
  - 同一运行目录里的 `runtime_env`（路径、sha256 和其中的变量，例如 `DURATION`）；
  - 运行目录下（最多三层）同一并发的 AIPerf 导出 `profile_export_aiperf.json`（`aiperf`、`aiperf_sha256`）。找到两份以上时一份也不用，原因写进 `missing`。
- `workload` 里写测量信息，都从这些文件读，不用 agent 写的：
  - `window`：AIPerf 自己的 `start_time`、`end_time`（客户端本地时间，原样）；
  - `node`：AIPerf 测的 endpoint 的主机（localhost 不算），读不到时取 runtime.env 的 `AIPERF_SERVER_URL`，多台用逗号连接；
  - `benchmark_duration_s`：AIPerf profiling 阶段的设定时长。InferenceX 的 fast 模式不管 `DURATION` 写多少都强制 1200 s，所以以 AIPerf 的为准；读不到、且不是 fast 模式时才用 `DURATION`；
  - `fast`：`runtime.env` 的 `AIPERF_EXPERIMENTAL_FAST`（ATOM kit 没有这个开关，它的 fast 就是短 `DURATION`）；
  - `duration_s`：AgentX 实测的时长；
  - 读不到的写进 `missing`（字段 → 原因），不编。
- 只有带 `agentx` 和 `agentx_sha256` 的正式结果能作判定输入；别的结果照常进台账，作参考。

**`ledger gsm8k`**：把 lm_eval 的 GSM8K 输出转成精度结果。

```
"$AGENT_SYS_PYTHON" -m agent_sys_v2.ledger gsm8k <lm_eval 输出目录> --out <results> --label <标签> --series <系列>
```

- 读一个目录里唯一的逐题记录 `samples_gsm8k_*.jsonl`（kit 的 `eval.sh` 带 `--log_samples`），数出 strict-match 下答对的题数和总题数；同一次运行的 `results_*.json` 正好一份时和它核对。
- 写 `results/<标签>.gsm8k.json`，不覆盖已有文件。格式 `infera.gsm8k/v1`：`correct`、`total`、`accuracy`、Wilson 95% 区间、答错的题号、其他 filter 的计数；`evidence` 里是逐题记录和汇总的路径与 sha256。
- 它不是结果文件，台账的曲线不读；`ledger check` 和 `ledger verify` 也校验它（verify 还按逐题记录复算）。
- `--series` 是它测的栈：版本 `v0`，或实验的系列。

**`ledger check`**：`"$AGENT_SYS_PYTHON" -m agent_sys_v2.ledger check <目录或文件> [--require-verdict]`，可以当验收检查用。有文件不对或一个结果都没有时退出码为 1 并列出原因。`--require-verdict` 要求每个候选都有判定或已作废（筛选的候选除外）。

#### A/B 判定

判定由 skill infera-experiment 的 `ab_judge.py` 写，核心在 `verdict.py`。

**输入**：
- `--write` 只接受 `--baseline-files`、`--candidate-files`：文件必须有 AgentX 来源、是正式测量，原始文件的摘要和重新转换出的指标都对得上，`--write` 时候选文件的 role 必须是 candidate。手填的数字（`--baseline 12010 11950`）只打印试算，不写。
- 两臂在同一个并发（并发按原始输出算，改结果文件里的字段藏不住）。project.yaml 写了 `baseline_series` 时，`--write` 还要求基线臂是它的复测、候选不是。
- **不许挑**：所用文件所在任务的 `results/` 顶层里，同一并发的有效正式结果，不论写的是什么系列和角色，都要作为输入（作废的除外；一个任务每个并发一个 A/B），否则拒绝。
- 同一份 AgentX 原始输出（sha256 相同）换个 label 转两次也只是一次测量：ab_judge 拒绝（退出码 2），`ledger verify` 判为对不上。

**口径**：ab_judge 读 project.yaml：指标、方向和 X 取 `ledger.primary`、`higher_is_better`、`ledger.x`；噪声带和接受线取 `ledger.noise_pct`、`accept_pct`（从 `$AGENT_SYS_PROJECT` 或结果文件所在的项目读）。命令行只能更严：
- `--accept-pct` 只能提高接受线；`--reproduce-pct`、`--max-arm-gap-min` 只能调小；
- 噪声带不能在命令行改：更宽的噪声带也会把拒绝变成噪声，不算更严；
- `--metric`、`--x-metric`、`--lower-is-better` 和 project.yaml 不一致时，`--write` 拒绝。

**判定函数**（`verdict.judge`，只看数字）：
- 噪声底 = max（固定部分，1.96 × 两臂差值的相对标准误）。固定部分 = max（`noise_pct`，本版本的复测噪声），见下面"噪声底"这道门。
- gain 在噪声底以内 → `noise`；损失超出噪声底 → `reject`。
- `accept` 要同时满足：gain 超过接受线和噪声底；X 不比基线差到超出 X 自己的噪声底（X 的固定部分是 `noise_pct`）；中位数和均值方向一致。
- 其余情况（介于噪声底和接受线之间、X 变差、中位数不一致）→ `rerun`。
- `--mechanism` 只是备注，不是条件。

**A/B 的门**（`gates.py`；ab_judge 和 `ledger verify` 调同一个入口 `verdict.judge_ab`）：数字判完后，依次过这些门。

1. **只认正式测量**：输入都要是正式测量，按运行目录判。筛选的点不算同伴，所以可以留在 `results/` 顶层。
2. **基线复现**：基线臂的均值和基线系列在同一并发的参照（`reference()` 再去掉本任务和其他实验的基线臂；参照点也按原始输出复算、只取正式的，没有 AgentX 来源的去掉并记在 `left_out`）的均值相差不超过 `ledger.reproduce_pct`（默认 5%）。否则判 `rerun`，备注写"基线臂没有复现基线系列（差 X%）"，以及只看数字会怎么判。参照在这个并发没有点时记"未核对"。
   - 依据：baseline 要求本地复测 ≤ 2%；v1 实测挨着测的两臂差 1.1%，隔 15 分钟加上邻居负载差 12%。5% 留出跨天、换节点的漂移，又能拦住邻居负载那种量级。
3. **测量时序**：同一节点上任意两个输入的窗口不能重叠（有一方不知道节点时，按"可能是同一节点"算）；按开始时间排好，每个窗口的开始距此前最晚的结束不超过 `ledger.max_arm_gap_min`（默认 60 分钟）。否则判 `rerun`。
   - 知道窗口的输入总是两两检查，个别输入没有窗口不影响其他的；不同节点上的窗口可以重叠，两臂分在几个节点上时备注写明。其余都合格、但有输入没有窗口（旧结果）时记"未核对"。不做 ABBA 配对。
   - 依据：同一节点上相邻两个 AgentX 点之间（重新部署加预热）实测 20–28 分钟；60 分钟能容纳一次重新部署，又排除"今天测基线、明天测候选"。
   - 已知空子：同一台机器在两个输入里分别写成 IP 和主机名时，会被当成两个节点。
4. **噪声底**：固定部分取 `noise_pct` 和本版本复测噪声中较大的一个。
   - 复测噪声 = 1.96 × √2 × 封版任务在同一节点的正式复测的相对标准差，即两次单点之差的 95% 区间半宽（和判定函数算 A/B 自身离散用的是同一个式子）。取 A/B 的并发，没有就取复测最多的并发；封版任务没有复测时依次退到上一个版本，直到 v0（v0 有 `perfcheck baseline` 保证）。都没有时这道门记"未核对"。
   - 固定部分高于接受线时，这次测量分辨不了接受线量级的变化，判定一律写 `resolvable: false`。其中判为 `noise` 的，备注自动写"this measurement cannot tell a change of the bar's size (noise floor X% > accept bar Y%) · 待集成"（备注是英文，只有"待集成"是 close 认的标记），界面显示"判不了 · 待集成"，留给 close 在集成曲线上判。不判 `rerun`：复测噪声是固定的，再测一对也降不下来。超出噪声底的变化照常判 accept、reject 或 rerun。
   - `resolvable` 按**项目的**接受线（快照里的 `accept_pct`）算：命令行 `--accept-pct` 调高的接受线只影响 accept 的门槛，不会把"判不了 · 待集成"变成普通的噪声。
5. **GSM8K**：数字判为 accept、基线复现和测量时序也过了时才查。
   - 本任务 `results/` 里的每一份 GSM8K 都参与（不许挑：不论写的是什么系列；改了 schema 或格式不对的直接判无法复算），和本版本的 GSM8K（封版任务测的、题数最多的一份，同题数取分数最高的）在共同的题上按 strict-match 比。
   - 候选系列里至少有一份不少于 200 题；每一份的逐题记录都要在本任务的工作区下，且不能是版本的那一份（不能把版本的运行换个 series 再转一次）。
   - Newcombe 95% 区间整个在 0 以下（显著变差）就改判 `reject`。
   - 没有候选的 GSM8K、或本版本没有 GSM8K 时，ab_judge 不写 accept；`ledger verify` 对 accept 也复算，没有就失败。

每道门的结果记在 `verdict.gates`：

| 门 | `status` 的取值 | 其他字段 |
|---|---|---|
| `reproduce` | ok / failed / unchecked | `off_pct`、`reference`、`reference_mean`、`reference_n`、`reproduce_pct`、`left_out`、`why` |
| `timing` | ok / failed / unchecked | `max_gap_min`、`nodes`、`note`、`why` |
| `noise` | ok / unchecked | `noise_pct`、`retest_pct`、`floor_pct`、`n`、`from`、`files`、`why` |
| `accuracy` | ok / worse / missing / unverifiable / unchecked（不是 accept 时不查） | `version`、`files`（每份的题数、两边的答对数、差值和 95% 区间）、`why`、`fix` |

failed 的门把判定改成 `rerun`，`worse` 把 accept 改成 `reject`；未核对的门写进备注，不改判定。

**判定里记下的输入**（`verdict.inputs`）：`metric`、`higher_is_better`、`x_metric`、`x_higher_is_better`、`baseline_series`、`baseline` 和 `candidate`（每份 `{file, agentx, agentx_sha256, value, x, measured: {window, node}}`，路径相对项目目录）、`noise_pct`、`accept_pct`、`min_duration_s`、`reproduce_pct`、`max_arm_gap_min`、`rules_from`（每个口径的出处，例如 `project.yaml` 或 `--accept-pct 4 (stricter than project.yaml: 3)`）。结果文件格式只做加法，仍是 `infera.result/v1`。

**作废只有两类**（`verdict.void.kind`，用 `ab_judge.py --void` 写进所列的 `--candidate-files` 和 / 或 `--baseline-files`）：
- `failed_requests`：由 AgentX 的请求数和错误数算出失败比例（错误 /（成功 + 错误）），超过 10% 才能作废；数字不让 agent 填。
- `other`：写 `reason` 和一个存在的 `evidence` 文件。

#### `ledger verify`：从原始输出复算

`"$AGENT_SYS_PYTHON" -m agent_sys_v2.ledger verify <目录或文件>` 是 experiment 类型的默认检查。

- 一份判定都没有（没有带 decision 或作废的结果）就失败。
- 每份 GSM8K 结果按逐题记录复算。
- 对每个带判定的结果（不论 role：把 role 改成 baseline 或 reference 藏不住判定，非候选带 decision 直接判为对不上）：
  1. 先做 `check --require-verdict` 的全部检查。作废的结果只查是不是两类之一，不需要 inputs。
  2. 要有 `verdict.inputs`，文件本身在候选输入里，同一个文件不重复出现。
  3. 每个输入：
     - 来源：结果文件在、没作废、有 AgentX 来源，记下的 `agentx` 路径和 sha256 与结果文件的 evidence 一致；原始文件在，sha256 一致；
     - **重新转换一遍**：指标和结果文件里的一致，`value`/`x` 和判定记下的一致，`workload.conc` 和原始输出一致，失败请求不超过 10%；
     - 测量信息：记下的 `runtime.env` 和 AIPerf 导出还在、sha256 一致，从它们读出的测量信息和结果文件里写的一致，是正式测量。
  4. 两臂在同一个并发；所用文件所在任务里同一并发的有效正式结果都在输入里，作废的属于两类之一；同一份原始输出没有算两次。
  5. 口径：指标、方向、基线系列和噪声带等于快照；接受线和正式时长不比快照低，复现容差和间隔上限不比快照松。
  6. 用同一个入口（`judge_ab`，含门）重算，`decision`、`vs_baseline_pct`、`noise_floor_pct`、`resolvable` 和每道门的 `status` 都要一致。没有 `gates` 记录的旧判定（inputs 里也没有门的口径）只比 `decision`、`vs_baseline_pct`、`noise_floor_pct`。
  7. 判为 accept 的，候选的 GSM8K 按逐题记录复算（没有、或逐题记录改过就失败）。
- **快照从哪来**：优先取 runner 交给检查的 `AGENT_SYS_RULES`；否则取正在执行检查的那一轮（`AGENT_SYS_CHECK_RUN`）或任务最近一轮的 launch.json；快照里缺的键（或者没有快照）取 project.yaml。只有来自某一轮的快照才检查基线系列（台账另用"相对 v0 判定"标出）。
- **口径在运行中被改**：命令行的 `ledger verify` 总会比较此刻 project.yaml 的口径和快照，不一样就失败（"project.yaml's ledger rules changed during the run"）。作为验收检查时这正是要的；但封版后 close 合法地改了 `baseline_series`，再手动复核以前的实验也会报这一条。台账页的复算不做这项比较。
- **失败分两类**，每条按"问题、上下文（context）、怎么改（fix）"写：
  - **判定和数据对不上**（the verdict does not match the data）；
  - **这份判定无法复算**（the verdict cannot be recomputed）：没有 AgentX 来源、原始文件没了、没有 inputs、文件读不了。

#### 结果文件格式

`infera.result/v1`（必填：`schema`、`label`、`metrics`）。下面是 `ledger agentx` 转换、`ab_judge.py --write` 判定之后的一个候选：

```json
{
  "schema": "infera.result/v1",
  "label": "r1-spf-c64",
  "time": "2026-10-24T10:00:00Z",
  "role": "candidate",
  "round": 1,
  "series": "r1-spf",
  "change": "--schedule-policy shortest-prefill-first (decode/prefill both)",
  "metrics": {"input_tput_per_gpu": 14120.5, "p90_intvty": 31.2, "p90_ttft": 2.1, "requests": 3120, "errors": 0},
  "workload": {"client": "agentx", "conc": 64, "duration_s": 3626.7, "benchmark_duration_s": 3600, "fast": false,
               "window": {"start": "2026-10-24T08:52:11.2", "end": "2026-10-24T09:52:37.9"}, "node": "10.0.0.1"},
  "config": {"hw": "mi355x", "image": "…", "model": "…", "framework": "atom", "topology": "…"},
  "verdict": {"status": "valid", "decision": "accept", "vs_baseline_pct": 15.4, "noise_floor_pct": 2.0,
              "notes": "+15.40% past the 3% bar and the floor 2.00%", "mechanism": "[M] queue-req ↓ 80%",
              "inputs": {"metric": "input_tput_per_gpu", "higher_is_better": true, "x_metric": "p90_intvty",
                         "x_higher_is_better": true, "baseline_series": "v0",
                         "baseline": [{"file": "workspaces/experiment/r1-spf/results/r1-spf-base-c64.json",
                                       "agentx": "workspaces/experiment/r1-spf/log/agentx/base/agentx_conc64.json",
                                       "agentx_sha256": "…", "value": 12236.1, "x": 31.0}],
                         "candidate": [{"file": "workspaces/experiment/r1-spf/results/r1-spf-c64.json", "agentx": "…", "agentx_sha256": "…", "value": 14120.5, "x": 31.2}],
                         "noise_pct": 2.0, "accept_pct": 3.0, "min_duration_s": 3600, "reproduce_pct": 5, "max_arm_gap_min": 60,
                         "rules_from": {"noise_pct": "project.yaml", "accept_pct": "project.yaml", "…": "…"}},
              "resolvable": true,
              "gates": {"reproduce": {"status": "ok", "off_pct": 0.4, "reference": ["workspaces/baseline/r1/results/v0-c64.json"]},
                        "timing": {"status": "ok", "max_gap_min": 26.0, "nodes": ["10.0.0.1"]},
                        "noise": {"status": "ok", "retest_pct": 1.2, "floor_pct": 2.0, "from": "baseline/r1: 2 formal measurements of v0 at C64 on 10.0.0.1"},
                        "accuracy": {"status": "ok", "version": "workspaces/baseline/r1/results/v0-full.gsm8k.json",
                                     "files": [{"file": "…/r1-spf-200.gsm8k.json", "questions": 200, "version": [192, 200], "candidate": [193, 200], "delta": 0.005, "ci95": [-0.037, 0.047]}]}}},
  "evidence": {"agentx": "workspaces/experiment/r1-spf/log/agentx/spf/agentx_conc64.json", "agentx_sha256": "…",
               "runtime_env": {"file": "…/runtime.env", "sha256": "…", "vars": {"DURATION": "3600", "…": "…"}},
               "aiperf": "…/aiperf_artifacts/profile_export_aiperf.json", "aiperf_sha256": "…"}
}
```

取值：`role` 是 `baseline` / `candidate`（默认）/ `reference`；`verdict.status` 是 `valid`（默认）/ `void`；`verdict.decision` 是 `accept` / `reject` / `noise` / `rerun`；作废时 `verdict.void` 是 `{"kind": "failed_requests", "failed_pct": 12.3}` 或 `{"kind": "other", "reason": "…", "evidence": "<文件>"}`。

### 5.7 skills 注入

- system.md 里有一段"skills 索引"，一行一个 skill：名字、描述、SKILL.md 的绝对路径。agent 用得上时自己去读。用法和经验索引相同，增加的上下文很小。
- **注入哪些**：`project.yaml` 的 `skills` 列表里的内置 skills，按列表的顺序（不写 `skills` 则注入全部内置 skills）；再加上项目自己 `skills/` 目录下的全部（与内置的同名时用项目自己的）。`skills` 里写了不存在的名字，launch.md 里会提醒，开始前检查报错（§5.5）。
- SKILL.md 的 frontmatter 和 Claude Code 原生 skill 的格式一致（`name`、`description`）；没有 description 的不算 skill。
- **system.md 各段的顺序**：agent_sys 约定（按 `protocol_language` 选中文版或英文版）、全局规则、项目规则、skills 索引、项目经验索引、全局经验索引。开头写明项目目录、`AGENT_SYS_HOME`、本任务的 prompt 和工作区。

### 5.8 API

- 路径都在 `/api` 下；任务的路径前缀是 `/api/projects/{p}/tasks/{类型}/{名字}`（下表写作 `…`）。
- 请求体是 JSON，字段类型不对返回 400。文件路径都相对于 `AGENT_SYS_HOME`。
- 拒绝的请求返回 `{"error": <给用户看的文字>}`（开始前检查拒绝时另带 `problems` 和 `project`）。状态码：400 输入有误，403 跨站或 Host 不对（§5.9），404 不存在，409 当前不允许（或冲突、已存在），422 文件无法解析，500 扫描项目或启动 tmux 失败。

| 方法与路径 | 说明 |
|---|---|
| `GET /api/home` | 首页：未归档项目的待确认、疑似卡住（`stalled`）、待审核、未查看、运行中、待开始数；今日花费（所有项目，含已归档）；项目和已归档项目；被占用的机器（每台的全部占用者 `holders`）；模板列表；新项目所在的目录 |
| `POST /api/resume` | 继续全部中断任务：`{tasks: [{project, task}]}`，只继续其中运行中断的；返回 `{resumed, errors}` |
| `POST /api/mark_read` | 标为已查看：`{tasks: [{project, task}]}`；返回 `{marked, errors}` |
| `POST /api/projects` | 新建项目 `{name, template, title}` |
| `POST /api/projects/{p}/types` | 新建类型 |
| `PUT /api/projects/{p}/lanes` | 泳道顺序 |
| `POST /api/projects/{p}/archive`、`…/unarchive` | 归档、取消归档；有运行中的任务时不能归档（409） |
| `POST /api/projects/{p}/start` | 开始运行：结束模板审核；开始前检查有错误时 409，带 `problems` |
| `GET /api/projects/{p}/board` | 看板：项目（含目标与台账摘要）、类型、任务、依赖边（含是否阻塞、`children` 的完成数）、机器表（每台机器的 `holders` 列表，空闲时为空）、名额；模板待审核时还有 `preflight`（开始运行会拒绝的问题） |
| `GET /api/projects/{p}/ledger` | 台账：设置、主指标、基线（`pin_missing`：钉住的基线没有可用结果）、最佳、行、系列、格式错误的文件 |
| `POST /api/projects/{p}/tasks` | 新建任务（排队或存为草稿） |
| `POST /api/projects/{p}/tasks/preview` | 新建任务表单的预览：prompt 和所选上游给出的依赖，名字能不能用 |
| `GET …` | 任务详情：上下游、草稿和 `drafts_problems`（全部纳入托管会拒绝的问题）、本轮的问题、最近一轮的检查、运行列表等。prompt 的错误（frontmatter、checks 解析不了）在 `warnings` 里；`upstream_changed` 是上游已更新的提示，`upstream_message` 是给 agent 的复核消息（agent 的语言） |
| `PUT …` | 修改任务配置（frontmatter 字段、prompt）；`send_to_agent` 时把新 prompt 告诉已运行过的任务，告诉不了时照样保存，并在 `notify_error` 里说明 |
| `POST …/rename` | 改名或改类型 |
| `POST …/actions/{action}` | `start`、`stop`、`resume`、`takeover`、`end_takeover`、`add_budget`、`cancel`、`join`、`join_children`（批量纳入子任务草稿，开始前检查有错误时 409，带 `problems`）、`approve`、`mark_read`、`delete`（移进 `.trash/`，返回 `{deleted, trash}`）；其他 action 返回 404 |
| `POST …/messages` | 你的消息：送进运行中的会话，或开始新一轮 |
| `GET/POST …/questions` | questions.md 表单：读、提交回答 |
| `GET …/runs/{run}/events?after=&before=&tail=` | 精简过的事件（长内容截断）；`GET …/runs/{run}/events/{seq}` 取一条全文 |
| `GET …/runs/{run}/launch` | 这一轮的 launch.md |
| `GET …/files?dir=`、`GET/PUT …/file?path=`、`GET …/raw?path=` | 工作区文件：列目录、读、写、原样下载 |
| `GET/PUT /api/rules` | 规则页（`?project=` 带上项目的） |
| `GET /api/knowledge`、`POST /api/knowledge/candidates/{adopt,to_rule,discard}`、`PUT /api/knowledge/entries`、`POST /api/knowledge/entries/delete` | 经验（`?project=` 带上项目的）：候选和经验库、采纳 / 改写为规则 / 丢弃、编辑和删除条目 |
| `GET/PUT /api/settings` | 设置（全局，或 `project` 指定的项目） |
| `GET /api/models` | 表单里可选的模型，以及别名、完整 id 在 agent 用的设置下（§5.9）指向什么 |
| `GET /api/events` | SSE：`changed {project, task}`（task 为 null 表示项目本身）、`run {project, task, run, seq}`（运行有新事件） |

### 5.9 安全与边界

**网络**
- **只监听 Unix socket**，默认 `/tmp/agent_sys-<uid>/agent_sys.sock`，目录权限 0700，只有你能连。浏览器通过 `ssh -L` 转发访问。
- **Host 校验**：转发之后，浏览器把 server 当成 localhost 访问，你电脑上的任何网页都能向它发请求。所以请求的 Host（去掉端口）必须是 `localhost`、`127.0.0.1`、`[::1]`，或环境变量 `AGENT_SYS_HOSTS`（逗号分隔）里的名字，否则返回 403（防 DNS rebinding）。
- **Origin 校验**：GET / HEAD / OPTIONS 以外的请求必须来自 server 自己的页面：有 `Origin` 时，它的 host:port 要等于 Host；没有 `Origin` 时，`Sec-Fetch-Site` 要是 `same-origin` 或 `none`。两个头都没有的请求（curl 之类）放行。
- **CSP**：响应带 `default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'`：脚本只从本站加载，组件可以用内联样式，图片可以是 data: 和 blob:，别的网站不能把界面嵌进 frame。agent 写的 Markdown 用 marked + DOMPurify 渲染，去掉脚本、表单、控件和样式。
- **`/raw`** 的响应不用上面的 CSP，而是带 `Content-Security-Policy: sandbox` 和 `X-Content-Type-Options: nosniff`：不管文件是什么后缀，都在没有脚本的不透明源里打开，不能调用 agent_sys 的 API。

**文件**
- **工作区文件**：路径相对于工作区，不接受绝对路径和 `..`。
  - 读取跟随符号链接，可以读到工作区以外（工作区常常链接到别处的数据，而只有你能连上 server）。
  - 写入先解析符号链接，真实路径必须在工作区内，否则拒绝；工作区内的符号链接保持是链接，写的是它指向的文件，文件权限不变。
  - `.agent_sys/` 不能通过界面修改。
  - 保存时带上开始编辑时的 mtime，文件在这期间被改过就拒绝（409）。
- **改名和删除**会移动工作区，所以运行中的任务、刚纳入托管（终端会话可能还没结束）的任务会被拒绝。

**agent 的权限**
- agent 以 `bypassPermissions` 运行，靠规则、检查和人的审核约束，不做 OS 隔离：agent 要操作集群、容器和仓库，隔离起来就做不了这些事。
- server 和 runner 去掉启动它们的 shell 里的 `CLAUDE*`、`ANTHROPIC*`、`IS_SANDBOX` 变量，避免在 Claude Code 会话里启动时覆盖用户配置。runner 还去掉 `TMUX`、`TMUX_PANE`（终端接管除外），免得 agent 随手一个 `tmux kill-server` 打到 agent_sys 的 tmux。发给 agent 的消息里不带任何凭据。

**agent 的 Claude Code 配置与你的隔离**（`claude_config.py`；`agent_sys.yaml` 的 `claude_config`，默认 `isolated`）

为什么要隔离：你的 `~/.claude` 里有你的 MCP server（`~/.claude.json`）、插件（`enabledPlugins`，以及 env 块里的 `CLAUDE_CODE_PLUGIN_DIRS`）、CLAUDE.md、skills 和 hooks。agent 以 `bypassPermissions` 运行，继承它们就等于不经确认地拿到你的外部系统写权限（实测一次运行带了 32 个 Atlassian MCP 工具：建 Jira issue、建和改 Confluence 页面），你的插件的 hook 也会在 agent 会话里生效。所以：
- **专用目录**：runner 和终端接管都以 `CLAUDE_CONFIG_DIR=$AGENT_SYS_HOME/claude` 启动 claude，加 `--strict-mcp-config`；setting sources 仍只有 user（即这个目录）。
- **settings.json**：每次启动按 `~/.claude/settings.json` 重新生成（文件权限 0600，目录 0700），只带下面两张白名单里的东西。launch.json 和 launch.md 只记键名，不记值。
- **`projects/`** 是指向 `~/.claude/projects` 的符号链接：resume、终端接管、`session_cost`、隔离之前的会话都照旧在那里找。链接是幂等的；那里已有非空的真目录或指向别处的链接时，不删、不改，拒绝开始并说明（409）。
- **`.claude.json`**（Claude Code 自己的状态）缺什么补什么：`hasCompletedOnboarding`、转发的 API key 已确认（`customApiKeyResponses`，和 Claude Code 一样存 key 的后 20 个字符）、项目目录已信任。全新目录里，交互式 claude 会依次弹出：选主题、确认 API key、登录、信任目录，没有 `skipDangerousModePermissionPrompt` 时再加上 bypass 警告，终端接管会卡在第一个上；补齐后都不弹（实测）。交互式 claude 仍会把官方插件市场的目录 clone 进来，但不启用其中的插件。
- **OAuth 登录**：`~/.claude/.credentials.json` 里有 Claude 的登录（`claudeAiOauth`）时，agent 的环境加 `CLAUDE_SECURESTORAGE_CONFIG_DIR=~/.claude`，原地读写（实测全新配置目录加这个变量后，`claude auth status` 显示 claude.ai 登录）。不用软链：软链读得到，但 Claude Code 写凭据（刷新 token）时不跟随软链（`O_NOFOLLOW`）；拷贝则会在第一次刷新后过期。文件里只有 MCP server 的 OAuth token（`mcpOAuth`；本机就是这样，是 Atlassian 的 token）时不设，agent 拿不到它们。
- **项目需要 MCP server 或插件时**，在 `agent.extra_args` 里写 `--mcp-config <文件>`（只能一个：SDK 只传一个文件，多个 server 写进同一个文件）、`--plugin-dir <目录>`（可以多个）。launch.md 的"Claude Code 配置"一节列出显式加的这些。
- **退路**：`claude_config: inherit` 退回旧行为：用你的 `~/.claude`，不加 `--strict-mcp-config`。设置页可以切换；launch.json 里没有 `claude` 字段的旧运行也按旧行为启动。

**env 白名单**（`FORWARD_ENV`）：
- 登录和模型路由：`ANTHROPIC_*`；`CLAUDE_CODE_USE_BEDROCK/VERTEX/FOUNDRY`、`CLAUDE_CODE_SKIP_*_AUTH`、`AWS_*`、`GOOGLE_*`、`CLOUD_ML_REGION`、`VERTEX_REGION_*`；`CLAUDE_CODE_OAUTH_TOKEN`、`CLAUDE_CODE_API_KEY_HELPER_TTL_MS`、`CLAUDE_CODE_CLIENT_CERT/KEY/KEY_PASSPHRASE`。
- 网络：`HTTP(S)_PROXY`、`NO_PROXY` 及小写形式、`NODE_EXTRA_CA_CERTS`、`SSL_CERT_FILE`。
- 网关和模型要的开关：`CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC`、`CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING`、`CLAUDE_CODE_MAX_OUTPUT_TOKENS`、`CLAUDE_CODE_SUBAGENT_MODEL`、`MAX_THINKING_TOKENS`、`API_TIMEOUT_MS`、`DISABLE_TELEMETRY`、`DISABLE_ERROR_REPORTING`、`DISABLE_AUTOUPDATER`、`DISABLE_NON_ESSENTIAL_MODEL_CALLS`、`DISABLE_PROMPT_CACHING*` 等。
- `IS_SANDBOX`：root 下跑 bypassPermissions 要它。
- 明确不转发：`CLAUDE_CODE_PLUGIN_DIRS`、`CLAUDE_CONFIG_DIR`、`CLAUDE_CODE_TMPDIR`、`PATH`、`TMPDIR`。不在白名单里的（如 `CLAUDE_CODE_DANGEROUSLY_SKIP_PERMISSIONS`，权限模式由 runner 自己定）一律不转发，launch.md 列出名字。
- agent_sys 自己的环境里已有的键不写进文件，让 agent 继承环境里的值：settings 的 env 块会盖过进程环境（实测），写进去就变成文件优先。`CLAUDE*`、`ANTHROPIC*`、`IS_SANDBOX` 已经从 agent_sys 的环境里去掉（见上），所以这些键总是按 settings.json 写进去；这条规则实际作用于 `AWS_*`、`GOOGLE_*`、代理之类。

**settings 键白名单**（`CARRIED`，逐个过了本机 settings.json 的顶层键）：

| 键 | 带不带 | 理由 |
|---|---|---|
| `env` | 按 env 白名单 | 见上 |
| `modelOverrides` | 带 | 把完整 id（`claude-opus-5-5` 等）映射到网关的名字。不带时按完整 id 选模型，请求里发的是原 id（实测），网关不认就路由不到 |
| `modelSettings`、`model` | 带 | 按模型的设置（如 effort）；任务没写模型时的默认模型 |
| `apiKeyHelper`、`awsAuthRefresh`、`awsCredentialExport` | 带 | 登录 |
| `skipDangerousModePermissionPrompt` | 固定为 true | 不论你怎么设：终端接管以 bypassPermissions 启动交互式 claude，否则会先弹危险模式确认 |
| `skipWorkflowUsageWarning` | 带 | 开了 ultracode 的终端接管不先弹用量提醒 |
| `autoCompactEnabled` | 带 | 它决定长会话上下文满了以后怎么办，会改变 agent 的行为；隔离之前 agent 一直按你的设置跑，保持不变 |
| `permissions` | 只带 `deny` | deny 只会收窄 agent 能做的事；`defaultMode` 由 runner 定（bypassPermissions），`allow`、`ask`、`additionalDirectories` 在这个模式下没有意义 |
| `enabledPlugins`、`extraKnownMarketplaces`、`hooks` | 不带 | 插件和 hook 正是要隔开的东西 |
| `theme`、`editorMode`、`tui`、`verbose`、`statusLine`、`outputStyle`、`askUserQuestionTimeout` | 不带 | 只影响你自己的界面 |
| `cleanupPeriodDays` | 不带 | runner 用 `--settings` 设为 365 天 |
| 其余 | 不带 | launch.md 列出名字；需要时加进 `CARRIED` |

`launch.models()`（表单里"别名和完整 id 指向什么"）读的是同一份设置：隔离时是生成给 agent 的设置，env 的值以环境为先；`inherit` 时是你的 settings.json。

### 5.10 环境变量

| 变量 | 谁读 | 作用 |
|---|---|---|
| `AGENT_SYS_HOME` | `agent-sys-v2` 命令 | home 目录：`--home` 优先，其次这个变量，再次 `~/agent_sys_home` |
| `AGENT_SYS_TMUX` | `agent-sys-v2 serve` | tmux server 的名字（`tmux -L`），默认 `agent_sys`；serve 把它带进 server 窗口，runner 的窗口开在同一个 tmux server 里。在正式 server 旁边再开一个（测试、演示）时换一个名字 |
| `AGENT_SYS_HOSTS` | server | Host 校验额外接受的名字，逗号分隔（§5.9） |
| `AGENT_SYS_PYTHON`、`AGENT_SYS_PROJECT`、`AGENT_SYS_TASK`、`AGENT_SYS_WORKSPACE`、`AGENT_SYS_MACHINES`、`AGENT_SYS_SKILLS` | agent、检查 | 运行 agent_sys 的 Python、项目目录、任务 id、工作区、机器（逗号分隔）、内置 skills 的目录（§5.4） |
| `AGENT_SYS_CHECK_RUN` | agent_sys 自己 | 检查进程的标记（值是 run 目录解析符号链接后的路径）：停止时据此找到并结束检查；`ledger verify` 据此找到这一轮的口径快照 |
| `AGENT_SYS_RULES` | 检查 | runner 交给检查的这一轮台账口径快照（JSON），`ledger verify` 和 `perfcheck` 优先用它 |
| `PYTHONSAFEPATH=1` | 检查 | 不把当前目录放进 `sys.path`，项目目录下同名的 `agent_sys_v2` 不会被优先 import（§5.4） |

agent 的环境里另外还有：
- 隔离时（§5.9）：`CLAUDE_CONFIG_DIR=$AGENT_SYS_HOME/claude`；有 Claude 的 OAuth 登录时还有 `CLAUDE_SECURESTORAGE_CONFIG_DIR=~/.claude`。
- `BASH_MAX_TIMEOUT_MS=86400000`：后台命令最长 24 h。
- `CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1`：内存紧张时不回收后台命令。
- `CLAUDE_CODE_EMIT_SESSION_STATE_EVENTS=1`：runner 靠会话状态事件判断一轮结束。
- `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`：不写用户的 auto memory。

## 6. Infera 性能工作包

### 6.1 模板 `infera-perf`

| 类型 | 用途 | 默认检查 |
|---|---|---|
| `intake` | 每轮开头的问卷和 brief（`review: true`）；可写项目的 `rules/project.md`、`project.yaml`、集群 skill，起草本轮的任务并给要上机的任务写 `machines` | report.md 非空；`perfcheck intake` |
| `survey` | 对标调研：InferenceX 参考曲线（`role: reference`）、配置差异表、合规清单；不占 GPU | report.md 非空；`perfcheck survey`；有 results 时 `ledger check` |
| `baseline` | 部署、过质量门（全量 GSM8K）、测 v0 曲线（`role: baseline`）、工作点同节点复测（噪声）、栈清单；快速验证标定可选 | report.md、`ledger check`、`perfcheck baseline`、stack.json |
| `profile` | 从便宜的层做起的瓶颈分析，产出 `candidates.yaml`（`review: true`；skill infera-profile；默认不用 ultracode） | report.md、candidates.yaml、profile.json |
| `plan` | 候选排序、预算和机器分配，起草 experiment 子任务和 close（`review: true`） | report.md 非空；`perfcheck plan` |
| `experiment` | 一个大杠杆或一组同层小杠杆：快速验证迭代，工作点单点 A/B（AgentX），写带判定的结果 | report.md 非空；`ledger verify` |
| `close` | 集成、整条曲线、封版、复盘，起草下一轮 intake；可写 `project.yaml` 的 version 和基线系列 | report.md 非空；有 results 时 `ledger check`；`perfcheck close` |
| `develop` | 代码改动：补丁、router、kernel，出 PR 或 overlay | report.md 非空 |
| `review` | 评审一个 PR、分支或结论（`review: true`） | — |

#### perfcheck：各类型的字段检查

`"$AGENT_SYS_PYTHON" -m agent_sys_v2.perfcheck <类型> "$AGENT_SYS_TASK"` 在项目目录下运行，每项打印一行 ok / FAIL，有一项失败就退出码为 1；轮次取任务名里的数字（`intake/r2`）。阈值取运行开始时的快照（§5.4）：close 有权改 project.yaml，改松了也不影响它自己的检查。

- **intake**：project.yaml 能解析，有目标、机器、轮次、台账口径（primary、x、working_point）；`rules/project.md` 没有剩下占位；问过问卷；本轮要上机的任务声明了机器；第 2 轮起本轮的 plan 已起草（报告写明项目结束时除外）。
- **survey**：results/ 里有对标结果（`role: reference`），或报告写明没有公开对标。
- **baseline**：这套测量能判 A/B。
  - 工作点（`{conc: C}` 就是 C；`{x: X}` 时取同一节点正式复测最多的并发）在同一节点上至少有 `ledger.min_repeats` 次 v0 的正式测量（按原始输出判，fast 和更短的不算），由此算出的复测噪声低于 `ledger.accept_pct`；
  - 有 v0 的全量 1319 题 GSM8K，逐题记录没改过。
  - 不满足时检查失败，提示 agent 把状态改为阻塞、交给你决定（每臂多测几次，或经你同意提高接受线），不带着这套测量进入实验。
  - `min_repeats` 默认 2：模板原来就是"同节点复测一次"，一次正式 AgentX 要 1–2 小时；2 次只有 1 个自由度，估出的噪声偏乐观的概率不小，项目可以改成 3。
  - `calibration.json` 不再是必需产物：fast 和 dstep 的标定只给实验迭代筛选用。
- **plan**：每个子任务声明了项目里有的机器，写了 `expect`（预期 Δ、快速验证级别 L0–L5、证伪条件）；有一个任务以 `children` 依赖等它的子任务，并声明了机器。
- **close**：`stack.json` 写明版本和是否封版。封版时：
  - 这个系列在 2 个以上并发有点，并且是 project.yaml 的 version 和台账的基线系列；
  - 有这个版本的全量 1319 题 GSM8K（有几份就每份都比，不许挑），和上一个有 GSM8K 的版本在共同的题上比，Newcombe 差值 95% 区间的下界不低于 `-quality.gsm8k_margin`（默认 0.02）；
  - 下一轮的 intake 已起草，并排在本任务之后。

**GSM8K 两档的分辨力**（p≈0.96，同题比较）：
- 200 题时，单臂 95% Wilson 区间约 ±2.8 个百分点，两臂差值约 ±4.2，要差 10 题（5 个百分点）才算显著变差，所以 200 题只当 accept 前的粗筛。
- 1319 题时差值约 ±1.5。2 个百分点的非劣界，只有在新版本比上一版低不超过约 0.5 个百分点时才证得出来，所以全量放在封版；这也是默认值 0.02 的依据：再小，两个一样好的版本也过不了。
- "差 0.01 以内"在两档都落在噪声里。

#### 模板的两个机制

- **模板自带任务**：`templates/<模板>/tasks/<类型>/<名字>.md` 在建项目时原样复制并排队（`skills/` 也复制，成为项目自己的 skill）。
  - 它们不会自己开始：project.yaml 写入 `template_review: true`，调度器不启动这个项目的任务，没有上游的任务等待原因是"等你审核模板"。
  - 你审核后在看板上点"开始运行"（`POST /api/projects/<p>/start`），或手动开始其中一个任务，这一行就删掉，之后和普通项目一样按依赖自动推进。
  - 两种开始都先做开始前检查（§5.5），有错误就拒绝。intake 要填的内容（`rules/project.md` 的占位符、空的 `machines`、目标、工作点）不在检查之列，模板审核时它们本来就是空的。
- **`children` 依赖**：`{task: X, kind: children}` 要等 X 完成，并且 `parent: X` 的任务全部完成或取消（草稿也算没完成；等待者自己和它的下游不计入）。
  - 看板上画成青色点划线，线的提示里写出子任务的完成数；等待原因写"等待 X 的子任务：…"；agent 的首条消息列出 X 的子任务和它们的报告。
  - X 的"全部纳入托管"纳入 X 起草的所有草稿：`parent: X` 的，以及直接依赖 X 的（评审覆盖除外）。草稿纳入托管时，frontmatter 没写的 `review` 和 `checks` 取类型的默认值。

一轮的任务图、各任务的产出、文件格式（实验子任务的 `expect`、close 的 `children` 依赖、`stack.json`、可选的 `calibration.json`）和问卷，写在模板自带的项目 skill `infera-round` 里。

#### 项目规则

项目规则 `rules/project.md` 写这个项目的背景（模型、引擎、kit、节点、负载、目标口径、工作点、SLA），以及实验纪律：
- 先诊断再选杠杆。一个子任务对应一个大杠杆，或一组同层、同验证方法的小杠杆；改动前写清预期机理、幅度和证伪条件。
- 以实际运行的配置为准，证明杠杆生效了再测。
- 证据标 `[M]` 实测或 `[I]` 推断。
- 判定只看 AgentX 正式测量（性能以 AgentX 为准，精度以 GSM8K 为准）：
  - 接受：ΔY 超过接受线（`ledger.accept_pct`，默认 3%）和噪声底，ΔX 不劣于噪声，中位数和均值方向一致，候选的 GSM8K 不显著变差；
  - 噪声底内算噪声，介于噪声底和接受线之间加跑。噪声底取 `ledger.noise_pct`（默认 2%）、本版本的复测噪声、这次 A/B 自身的离散三者中最大的；
  - 前两者中较大的高于接受线时，噪声底内的结果判"判不了 · 待集成"，留给 close；
  - 基线臂没复现基线系列、两臂窗口重叠或隔得太久，判 rerun；
  - 机理指标、fast 模式、dstep 用于筛选和解释，不是接受条件；
  - 口径只由用户改，命令行只能更严；
  - 用 X 换 Y 的改动在等 X 下比较（工作点 Δ）。
- 作废只有两类：失败请求超过 10%（从 AgentX 计数算）；其他原因（别人的负载、HSA out-of-resources、被抢占……）写原因和证据文件。
- 不动别人的容器和 GPU；长任务在远端以自包含的执行链运行，并带守护。

### 6.2 内置 skills

| skill | 内容 |
|---|---|
| `infera-deploy` | kit 的启停；VRAM 等待和外来容器检查；生效配置审计（`docker inspect`、`/get_server_info`），找出没被转发的 knob；守护规则 |
| `infera-bench` | AgentX 全量和 fast 模式、固定 ISL/OSL 负载；每个点的固定产物；写 `infera.result/v1`；测量陷阱 |
| `infera-quality-gate` | 拒绝在 forced acceptance 下跑；GSM8K 是唯一必须过的精度门（`ledger gsm8k` 转换，两档按区间判）；smoke、乱码和长上下文探针、MTP 真实接受、RDMA 不回退 TCP、PD 交接是排查用，不作为门 |
| `infera-profile` | 一轮的 profile：从便宜的层做起（时间账、资源利用），只对瓶颈层做 kernel 或 CPU；给 plan 的 `candidates.yaml`；引擎 playbook（目前只有 ATOM 的一份，尚不完整；SGLang、vLLM 还没有） |
| `infera-diagnose` | 瓶颈分类（prefill 饱和、decode 步长、KV 容量、传输、CPU 路径、JIT、正确性）；每一类的观测手段和候选杠杆 |
| `infera-experiment` | 单变量 A/B 协议、两类作废和判定规则、`scripts/ab_judge.py`（只用 AgentX 正式结果；口径取 project.yaml、命令行只能更严；A/B 的门；噪声底；噪声不会判成接受；同时看 X；`--write` 把判定和 `inputs` 原子写入每个候选文件）、用 `ledger verify` 自查 |
| `infera-hygiene` | 共享节点卫生：不删别人的容器，NFS、TMPDIR、HOME 陷阱，VRAM 释放，抢占 |

另有 `infera-perf` 模板自带的项目 skill `infera-round`（§6.1）。

skills 的写法：
- 确定性的事交给脚本，判断交给 agent；
- 写明"怎么证明它真的生效了"（目前 deploy、experiment、quality-gate 有这一节）；
- 不复制仓库里已有的脚本，而是指过去（例如 `repos/Infera/examples/<kit>/engine/up.sh`）。

## 7. 可观测与可控清单

| 你想知道或做的事 | 在哪里 |
|---|---|
| 现在哪些事需要我 | 首页待确认（含验收未通过、闲置占用、运行中疑似卡住）；顶栏和标签页标题上的数字 |
| 某个 agent 正在干什么 | 卡片上的"正在执行：<命令>"，超过一分钟加上"已运行 N 分钟（超时 M 分钟）"；运行输出实时流 |
| 它是不是卡住了 | 卡片和抽屉的"疑似卡住"：哪一种、多久了（§7.1） |
| 它在等什么 | 运行中 · 等待中：后台命令的描述、已等多久；待开始的卡片写明等待原因 |
| 它拿到了哪些规则、skills、经验 | 运行输出里的"本次注入内容"（launch.md） |
| 它带了我的哪些 Claude 配置 | launch.md 的"Claude Code 配置"：转发和没转发的 env 键、带和没带的 settings 键、显式加的 MCP 和插件 |
| 花了多少 | 抽屉头部的花费/预算、任务列表的花费列、待审核卡片的本轮花费；顶栏和首页的今日花费 |
| 哪台机器被谁占着、是否闲置 | 看板机器条；首页机器卡片 |
| 优化走到哪了 | 台账：各系列的 Y–X 曲线和工作点 Δ，基线和最佳，每个结果的判定 |
| 结论可信吗 | 检查 ✓/✗；结果的判定和"判定未核实"标记；报告里标 `[M]/[I]` 的证据；报告审核 |
| 让 agent 改方向 | 运行输出底部的输入框（运行中也可以） |
| 立刻停下 | 停止（打断当前命令，检查也一起停） |
| 亲自上手 | 在终端中接管（同一会话），退出后自动接回 |
| 控制花费 | 每个任务的预算；追加预算 |
| 批准 agent 起草的计划 | 父任务抽屉：全部纳入托管 |

### 7.1 疑似卡住

运行中的会话可能停住：模型调用挂起没有回应，或者前台命令本该被结束却一直没有返回。agent_sys 只看运行的事件来判断，**不看运行了多久**：v1 的经验是，按总时长设上限（300 s、1800 s）都误杀过健康的运行，还让 GPU 一直被占着。

判断只提醒：
- 卡片标"疑似卡住"，写明是哪一种、已多久；
- 任务进首页和看板"待处理"的待确认列表（排在最前，主按钮是发消息、停止、在终端中接管），项目列表、顶栏和标签页标题的待确认数也算上它；
- 任务状态不变，不自动停止，也不自动重发消息；
- 提醒的条件不再成立时自动消失：`silent` 在有新的前台事件后消失，`timeout` 在那个工具返回后消失。

**两种情况**（`events.Follow.stall`，server 的 Monitor 每 0.5 s 对运行中的任务算一次，写进 `detail.stall`）：

| 种类 | 条件 | 卡片上 |
|---|---|---|
| `silent` 没有事件 | 会话 running，前台没有带超时的工具在跑，最后一个"前台事件"距今超过 `stall_minutes` | 没有前台工具："模型调用无响应，已 N 分钟没有任何事件"；前台是子 agent："前台子 agent 已 N 分钟没有任何事件"；其他没有超时的工具："<工具> 已 N 分钟没有返回，也没有任何事件" |
| `timeout` 超时未返回 | 前台工具已运行的时间超过它自己的超时再加 5 分钟宽限 | "前台命令超时未返回（Bash 已运行 N 分钟，超时 M 分钟）" |

**每类前台工具归哪条**：

| 前台工具 | 规则 | 超时 |
|---|---|---|
| `Bash` | 超时未返回 | 输入的 `timeout`（毫秒）；没写时取 `BASH_DEFAULT_TIMEOUT_MS`：launch.json 的 env、server 的环境、agent 的 Claude 配置目录和你的 `~/.claude` 的 settings.json 里能找到的最大值，并且不低于 120 s（Claude Code 自己的默认值）。取最大：在没看到的地方设了更大的默认值时，健康的命令不会被误报。Claude Code 到超时就把命令移到后台并立刻返回 |
| `TaskOutput` | 超时未返回 | 输入的 `timeout`，默认 30 s |
| `Agent` / `Task`（前台子 agent） | 没有事件 | 没有超时概念，健康时可以连续跑几十分钟；子 agent 自己的帧（带 `parent_tool_use_id` 的消息和工具结果、它的 `task_progress`）算作事件 |
| 其他（`Read`、`Write`、`Edit`、`Grep`、`WebFetch`、MCP 工具……） | 没有事件 | 没有超时参数 |

子 agent 里在跑的带超时的命令同样按它的超时判，期间子 agent 没有帧也不算卡住。

**什么算"前台事件"**（`Follow.moves`）：CLI 的帧都算，除了：
- 前台工具每 30 s 一次的心跳 `tool_progress`：CLI 不管工具在做什么都发，子 agent 的模型调用挂住时它照样来；
- 后台的事：`background_tasks_changed`，后台任务的 `task_*`，后台子 agent 和它的工具的帧（含它被拒的命令 `permission_denied`）；
- git 的变化 `vcs_state_changed`、`code_change_published`：可能是后台子 agent 做的；主会话自己的命令引起的，也只比命令晚几秒；
- 不落盘的帧（`thinking_tokens`、`ui_status`、`ui_panes`，§5.1）：所以一次只输出思考内容的长模型调用也显得没有事件；
- runner 自己的事件只算 `connected`（CLI 启动后一直没有输出也能发现）和验收检查的开始、结束。你发的消息不算：发给一个挂住的会话，提醒不应该消失。

**不判断的时候**：
- 会话空闲（后台等待、一轮结束）；
- 一轮已经以 `result` 结束、后台子 agent 还在跑：这时会话状态仍是 running，直到下一轮的 `init`、`session_state_changed` 或消息帧（`api_retry` 不算新一轮，它不带是谁的）；
- 正在跑验收检查（它们有自己的超时）；
- 终端接管；
- 在别的主机上的运行（只显示，时钟也不是本机的）；
- `stall_minutes: 0`。

**配置**：`agent_sys.yaml` 的 `stall_minutes`，默认 15，可以是小数，0 关闭；设置页的全局设置里也能改。宽限 5 分钟是固定值。

**只能发现"停了"**：在原地打转（反复重试、反复跑同一条命令）的会话一直有事件，不会被标出来。API 出错时 CLI 自己重试，每次重试的 `api_retry` 帧算作事件。

**阈值怎么来的**：`tools/stall_calibrate.py` 把 events.jsonl 逐帧喂给 `Follow`，按会话当时在做什么统计安静的时长，并数出 `Follow.stall` 在不同阈值下会给几次提醒。用本机同一 SDK 帧格式的记录标定，去掉回放和重复的副本后是 121 个健康运行、约 50 小时：

| 安静的时长（s） | 次数 | p50 | p99 | p99.9 | 最长 |
|---|---|---|---|---|---|
| 模型调用（running，前台没有工具） | 11603 | 2.5 | 132 | 263 | 292 |
| 前台有带超时的工具 | 6428 | 0.1 | 147 | 379 | 595 |
| 前台 Read | 411 | 0 | 0.8 | 265 | 265 |
| 一轮结束、后台子 agent 还在跑 | 186 | 0 | 1376 | 1987 | 1987 |
| 后台等待（会话空闲） | 25 | 309 | 15058 | — | 15058 |

- 最长的几次模型调用（250–290 s）几乎都以 CLI 的 `api_retry` 结束：请求挂住后，CLI 大约 4–5 分钟后自己重试。
- 4862 次前台 Bash 里，37 次跑到了超时，全部在超时后 0.4 s 内返回（移到后台）；最长 598 s，没有一次超过它的超时。
- 按上面的规则，5、10、15、20、30 分钟的阈值在这 121 个运行上都是 0 次提醒。如果不在 `result` 后停止计时，15 分钟时会有 3 次误报（一轮结束后，后台子 agent 跑了 11–33 分钟）；如果把后台子 agent 的帧算作事件，后台有子 agent 时主会话的模型调用挂住就发现不了（有测试）。
- 本机 682 个 Claude Code 会话记录（transcript）里另有 68833 次模型调用的间隔：p99.99 是 434 s，超过 900 s 的只有 1 次（1217 s，只输出了 1344 个 token，是一次挂住的调用）。v1 的实测是健康会话相邻记录最多隔 257 s，卡死的那次 65 分钟没有记录。默认 15 分钟是健康最长间隔的 3 倍以上，又远小于 65 分钟。
