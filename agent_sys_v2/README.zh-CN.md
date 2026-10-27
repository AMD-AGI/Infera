[English](README.md) | 中文

# agent_sys v2：Infera 性能工程的 agent 工作台

一名工程师在网页上指挥一组 Claude Code agent，找出并消除 Infera 部署的瓶颈。瓶颈可能在拓扑与并行、引擎参数、CPU 路径、通信、KV 传输或 kernel；每个结论都要有实测结果，每个 agent 都看得见、管得住。

- **任务 = prompt + 工作区。** agent 自己决定查什么，skills 提供可靠的工具。任务图在工作中长出来：agent 起草子任务，你一键批准整组计划。
- **核验证据，而不是限定步骤。** agent 说完成后，由系统执行验收检查，失败就把输出发回 agent。结果用统一格式写，进台账，显示基线、最佳值和曲线。
- **看得见，管得住。** 每次运行都有实时输出，注入了什么、花了多少、在等什么都看得到。可以发消息、停止、在终端接管、追加预算、审核。一台机器同一时间只给一个任务。

它和 [`../agent_sys`](../agent_sys)（v1，任务包运行时）并列，不改动 v1。两者可以装在同一个环境里：v1 的命令是 `agent-sys`，v2 的命令是 `agent-sys-v2`。产品、前端、后端的设计和规则见 [`docs/design.zh-CN.md`](docs/design.zh-CN.md)。

## 前提

- Linux、Python 3.10+、`tmux`。
- 能正常使用的 Claude Code 配置 `~/.claude/settings.json`（认证、网关、模型映射）。运行用的是 `claude-agent-sdk`（版本固定）自带的 `claude` CLI，加上你的用户配置，不需要别的设置。

## 安装

在 Infera 仓库根目录，进入任意虚拟环境：

```bash
python3 -m venv ~/.venvs/agent-sys-v2 && . ~/.venvs/agent-sys-v2/bin/activate
pip install -e agent_sys_v2            # 也可以 uv pip install -e agent_sys_v2；要跑测试加 [dev]
agent-sys-v2 --version
```

最好从单独的 checkout（或 wheel）安装，不要和项目任务要改的 Infera 仓库共用一个 checkout：验收检查会调用 agent_sys_v2 自己的模块，运行开始时记下它们的摘要；运行期间在那个 checkout 里 `git checkout` 或 `pull`，检查就会失败。

## 启动

```bash
agent-sys-v2 serve
```

- server 放在 tmux 窗口里（`tmux -L agent_sys` 的 `server` 窗口），**只监听 Unix socket** `/tmp/agent_sys-<uid>/agent_sys.sock`，所在目录只有你能进入。启动时会打印位置和下一步该敲的命令。
- 在你自己的电脑上转发端口，然后用浏览器打开：

  ```bash
  ssh -N -L 8765:/tmp/agent_sys-<uid>/agent_sys.sock <你>@<这台机器>
  # 浏览器打开 http://localhost:8765
  ```

- 看日志：`tmux -L agent_sys attach -t agent_sys:server`。
- 停止：在这个窗口里按 Ctrl-C，或者 `tmux -L agent_sys kill-window -t agent_sys:server`。重启 server 不影响正在跑的任务：每次运行都是同一个 tmux 里的独立进程，server 起来后会从文件里接上它们。
- 其他选项：
  - `--foreground`：在当前终端运行；
  - `--home DIR`（或环境变量 `AGENT_SYS_HOME`）：换一个 home；
  - `--sock PATH`：换一个 socket；
  - `AGENT_SYS_TMUX=<名字>`：换一个 tmux server，用于在正式 server 旁边再开一个。

所有东西都在 `AGENT_SYS_HOME` 里，默认 `~/agent_sys_home`，第一次使用时自动创建：

```
agent_sys.yaml  rules/  knowledge/                     全局设置、规则、经验
projects/<p>/project.yaml                             目标、机器、台账主指标、skills、agent 默认值
projects/<p>/rules/project.md                         项目背景，每次运行都注入
projects/<p>/tasks/<类型>/<名字>.md                     任务 prompt（含 frontmatter）
projects/<p>/workspaces/<类型>/<名字>/                  progress.md、report.md、results/*.json、.agent_sys/runs/
projects/<p>/repos/                                   agent 要改的代码（不扫描）
```

旧版本建的 home 把项目放在 `campaigns/` 下，现在不再读这个目录。先停掉它的 server，再改一次名：`mv <home>/campaigns <home>/projects`。

## 第一个项目

1. 在首页**新建项目**，模板选 **Infera 性能优化**（`infera-perf`）；或者用命令行 `agent-sys-v2 new glm52-agentx --template infera-perf --title "GLM-5.2 4P4D AgentX"`。
   模板自带第 1 轮的 5 个任务，建项目时就排好队：`intake/r1` → `survey/r1` ∥ `baseline/r1` → `profile/r1` → `plan/r1`。再把 Infera clone 到 `projects/<p>/repos/Infera`：agent 在项目目录下工作，skills 里引用的是 `repos/Infera/examples/<kit>`。
2. 这时还不会开始：看板顶部显示**待审核模板**，按执行顺序列出这 5 个任务。逐个打开，检查（可以直接修改）prompt、机器和验收检查，再看一遍项目规则和设置；确认后点**开始运行**（或者在某个任务里点**开始**）。之后的任务按依赖自动推进。
3. `intake/r1` 开始：先预调研（权重、InferenceX、节点），再给你一张预填好的问卷（**回答 N 个问题**；没填的行按建议处理）。拿到回答后它写好项目的 `rules/project.md`、`project.yaml` 的目标、机器和工作点，出一页 brief 等你审核。审核通过前下游一律不开始。
4. 对标调研（InferenceX 参考曲线，不占 GPU）和基线（v0 曲线、噪声、快速验证标定）并行，然后是 profile（从便宜的层做起，产出 `candidates.yaml`，需要你审核）。
   - 在**运行输出**里看进展；用底部输入框给 agent 发消息，运行中也可以；需要时点**停止**，或**在终端中接管**。
5. `plan/r1` 给候选排序、分配本轮的 GPU 预算和机器，起草实验子任务和 `close/r1`。审核通过后在它的详情里点**全部纳入托管**，实验和收尾任务一起排队。
   - 每个实验占用自己的机器，其他任务等机器空出来再开始；
   - `close/r1` 用 `kind: children` 依赖等 plan 的子任务全部完成或取消，再集成本轮接受的改动、测整条曲线，比 v0 好就封版 v1，并起草下一轮的 `intake/r2`（带建议的增量问卷）。
6. **台账**页把每个系列画成 Y–X 曲线（我们的、基线、对标），给出各系列在工作点的值和 Δ。agent 说完成了但验收检查没过的任务，会出现在**待确认**里，附带检查的输出。

任务文件示例（`depends_on` 的 kind：`input`、`code` 等上游完成，`children` 等上游和它的全部子任务完成；`review`、`subtask` 不阻塞）：

```markdown
---
title: SPF scheduling at C64
depends_on: [baseline/b0]
machines: [smc-a16-19]            # 运行中和待确认时占用
checks:                           # agent 把 progress.md 标为已完成后执行；失败就发回 agent
  - test -s "$AGENT_SYS_WORKSPACE/report.md"
  - '"$AGENT_SYS_PYTHON" -m agent_sys_v2.ledger verify "$AGENT_SYS_WORKSPACE/results"'
check_retries: 2
agent: {model: opus, effort: high, budget_usd: 50}
---
## 任务描述
……
```

结果文件是 `results/<标签>.json`，格式为 `infera.result/v1`（见 [`src/agent_sys_v2/skills/infera-bench/SKILL.md`](src/agent_sys_v2/skills/infera-bench/SKILL.md)）。A/B 判定由 `src/agent_sys_v2/skills/infera-experiment/scripts/ab_judge.py` 写入，只用从 AgentX 输出转换的结果；`ledger verify` 从原始输出复算。

## 不调用模型先看看

```bash
python agent_sys_v2/tools/demo.py /tmp/$USER/agent_sys_demo
AGENT_SYS_TMUX=asys-demo agent-sys-v2 serve --home /tmp/$USER/agent_sys_demo --sock /tmp/agent_sys-$(id -u)/demo.sock
ssh -N -L 8766:/tmp/agent_sys-<uid>/demo.sock <你>@<这台机器>     # 在你的电脑上；浏览器开 http://localhost:8766
```

这是一个演示项目：3 台机器，任务处在各种状态，有台账，也有一组起草好的计划。它的 `project.yaml` 设置了 `agent.replay`，每次运行都回放一段录好的会话（`tests/fixtures/events/`），不调用模型。用 `tmux -L asys-demo kill-server` 停止。`tools/demo.py` 只会替换它自己生成的目录：目标目录不是空的、又不是之前生成的演示 home 时，它拒绝执行。

## 开发

```bash
pip install -e "agent_sys_v2[dev]"
cd agent_sys_v2 && python -m pytest tests -q     # 3–4 分钟，需要 tmux；测试都不调用模型
node --test src/agent_sys_v2/web/test/*.test.js  # 前端的纯函数和英文词典
ruff format --check . && ruff check .            # 仓库的 ruff 配置
```

| 路径 | 内容 |
|---|---|
| `src/agent_sys_v2/server.py`、`cli.py` | REST + SSE（把文件扫描进内存、推送变化、提供网页）；`agent-sys-v2` 命令 |
| `src/agent_sys_v2/model.py`、`scan.py`、`ops.py` | 文件对应的数据模型（任务、依赖边、state.json）；扫描项目、发现变化；写操作 |
| `src/agent_sys_v2/sched.py`、`machines.py` | 调度：依赖、名额和机器 |
| `src/agent_sys_v2/runs.py`、`runner.py`、`launch.py`、`tmux.py` | 每个任务一个 Claude Code 会话，每轮一个 runner 进程（tmux 窗口）；一次运行注入什么（system.md、首条消息、launch.md） |
| `src/agent_sys_v2/events.py`、`procs.py`、`replay.py` | 运行事件（界面怎么读）；运行的进程；不调用模型、回放录好的会话 |
| `src/agent_sys_v2/checks.py`、`ledger.py`、`perfcheck.py`、`skills.py` | 验收检查；结果台账；`infera-perf` 模板的字段检查；skills 索引 |
| `src/agent_sys_v2/knowledge.py`、`questions.py`、`files.py`、`i18n.py` | 规则和经验；questions.md 表单；工作区文件；英文翻译 |
| `src/agent_sys_v2/web/` | 网页界面：Preact + htm，不需要构建（[README](src/agent_sys_v2/web/README.md)） |
| `src/agent_sys_v2/templates/` | 每次运行注入的约定（中 / 英）和项目模板 `blank`、`infera-perf` |
| `src/agent_sys_v2/skills/` | Infera 性能工作的 skills（[README](src/agent_sys_v2/skills/README.md)） |
| `tests/` | pytest；运行用录好的会话回放 |
| `tools/demo.py` | 上面的演示 home |
| `docs/design.zh-CN.md` | 设计文档 |

## 常见问题

- 提示 *socket 所在的目录其他用户也能进入*：把 `--sock` 指到一个只有你能进入的目录。
- 提示 *已有 server 在运行*：一个 `AGENT_SYS_HOME` 只能有一个 server，否则两个调度器会把同一个任务启动两次。
- 待开始的任务迟迟不开始：卡片上写着原因，可能是项目已归档、prompt 的 frontmatter 或 checks 有错误（改正后才会开始）、上游未完成（`children` 依赖还要等上游的子任务）、模板待审核（在看板上点**开始运行**）、关闭了自动开始、机器被别的任务占用，或名额已满。
- 运行用的是 `~/.claude/settings.json`。server 会去掉启动它的 shell 里的 `CLAUDE*` / `ANTHROPIC*` 变量，所以即使在 Claude Code 会话里启动 server，用的也是你的用户配置。
- v1 也读 `AGENT_SYS_HOME`（作为它的状态前缀）。如果你为 v1 设置了它，v2 会在同一个目录下建自己的子目录，两者互不干扰。
