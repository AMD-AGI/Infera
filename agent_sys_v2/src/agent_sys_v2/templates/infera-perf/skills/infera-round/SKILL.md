---
name: infera-round
description: 本项目一轮的做法：任务图、各任务的产出和文件格式（实验子任务的 expect、close 的 children 依赖、stack.json、可选的 calibration.json）、问卷模板，以及验收检查查什么
---

# infera-round：一轮怎么走

## 任务图

```
intake/rN（问卷 + brief，review）─┬─► survey/rN（对标，无 GPU）─────────────┐
                                  └─► baseline/r1（GPU，只在第 1 轮）─► profile/rN（review）─┴─► plan/rN（review）
plan/rN 起草：experiment/rN-<杠杆或包> × N（子任务，parent: plan/rN）
              close/rN（不是子任务；depends_on: [{task: plan/rN, kind: children}]）
close/rN 起草：intake/r(N+1)（depends_on: [close/rN]）
```

- 第 1 轮的 intake、survey、baseline、profile、plan 由模板在建项目时预置并排队，我审核后点"开始运行"才开始。之后每轮的任务由 intake/rN 按问卷回答起草（survey/rN、plan/rN，需要时 profile/rN 和 baseline/rN），close 的集成曲线就是下一轮的基线，通常不再重做 baseline。
- `kind: children` 的依赖：plan 完成、且 `parent: plan/rN` 的任务全部完成或取消后，close 才开始。提前收尾 = 取消剩下的子任务。
- 你起草的任务都是草稿；我在你的任务抽屉里点"全部纳入托管"，你的子任务和直接依赖你的草稿一起排队。
- **plan、intake、close 起草完任务后，标完成之前先跑整体检查**：`"$AGENT_SYS_PYTHON" -m agent_sys_v2.cli check "$AGENT_SYS_PROJECT" --drafts-of "$AGENT_SYS_TASK"`。它一次列出你起草的草稿（和"全部纳入托管"查的一样）以及 project.yaml 的问题：依赖或 parent 写错、阻塞依赖成环、frontmatter 或 checks 解析不了、machines 不在 project.yaml 里、project.yaml 的 skills 不存在、checks 里 `agent_sys_v2.<模块>`、`perfcheck <类型>`、`ledger <子命令>` 或 `$AGENT_SYS_SKILLS/`、`$AGENT_SYS_PROJECT/` 下的脚本不存在。每条带改法，名字写错时给出最接近的名字。有 `error`（退出码 1）就改到通过：同样的检查会挡住"全部纳入托管"和"开始运行"。`warn` 看情况处理，在 report.md 里说明。不带 `--drafts-of` 时查项目里所有未完成的任务；别的任务的问题不要替它改，写进 report.md 告诉我。
- 我介入的地方每轮有三处：intake 的问卷和 brief、profile 的审核、plan 的审核（候选、预算、机器）。其余选择按默认继续，写进"替你做的决定"。

## 各任务的产出（验收检查会查带 ✓ 的）

| 任务 | 产出 |
|---|---|
| intake | ✓ questions.md（问卷）；✓ 项目的 `rules/project.md`（占位全部换掉）；✓ `project.yaml` 的 goal、machines、round、ledger（primary、x、working_point）；✓ 本轮要上机任务的 machines；`skills/cluster-<名>/SKILL.md`（集群操作手册）；report.md（一页 brief 和校验记录）；第 2 轮起 ✓ 本轮的 plan 草稿 |
| survey | IX 参考曲线进台账（✓ `role: reference`，`series: ix:<条目>`，或报告写明"无公开对标"）；配置身份表；**配置差异表**（每项：我们的值、对方的值、预期影响）；合规清单 |
| baseline | 环境预检；✓ 全量 1319 题 GSM8K（`ledger gsm8k --series v0`）；B1 曲线（`role: baseline`，`series: v0`，4–7 个并发）；✓ 工作点同一节点 ≥ `ledger.min_repeats` 次正式复测，复测噪声低于 `ledger.accept_pct`（否则阻塞、交给我）；✓ `stack.json`（v0）；常开的系统采样；可选 `calibration.json` |
| profile | report.md、✓ `candidates.yaml`、✓ `profile.json`、`verify.md`（skill infera-profile） |
| plan | 候选排序表（来源、预期区间、成本、验证级别）；本轮 GPU 预算和机器分配；实验子任务草稿（✓ 各自的 machines 和 expect）；✓ close/rN 草稿 |
| experiment | 生效证明；快速验证记录；候选栈 200 题 GSM8K（accept 必需）；L4 工作点 A/B（AgentX 正式测量，`ledger agentx` 转换；基线臂复现基线系列，两臂同节点不重叠、间隔有上限）；✓ 判定写进结果（`ab_judge.py --write`），验收时从原始输出复算（`ledger verify`，含 GSM8K） |
| close | 集成镜像；✓ 全量 1319 题 GSM8K，不比上一版差超过 `quality.gsm8k_margin`；L5 曲线（`series: vN`）；工作点同节点复测（下一轮的噪声底）；✓ `stack.json`；预测与实测对照；复盘和下一轮建议；✓ intake/r(N+1) 草稿 |

## 文件格式

起草任务时注意：任务 prompt 里写到的 `@<类型>/<名字>`、`tasks/<类型>/<名字>.md`、`workspaces/<类型>/<名字>/` 都会被当成"依赖它"。只这样引用上游任务；提到下游任务（例如 intake/r2 的 prompt 里提到本轮的 plan）时写"本轮的 plan"之类的文字，不写成路径，否则会形成循环依赖，任务一直等待。

**实验子任务**（plan 起草，`tasks/experiment/r<N>-<名>.md`）：

```yaml
---
title: SPF 调度 + tokenizer workers
parent: plan/r1
summary: prefill 队头阻塞：shortest-prefill-first 和 tokenizer ×4 打包验证
machines: [<plan 分配的节点>]
expect:
  delta_pct: [2, 4]         # 工作点 ΔY 的预期区间（%），或一个数
  level: L2                 # 迭代用的快速验证级别 L0–L5；最后统一测 L4 工作点
  falsify: "prefill 队列 p90 没有下降 30%"   # 什么结果说明假设不成立
---
```

**close**（plan 起草，`tasks/close/r<N>.md`）：

```yaml
---
title: 第 1 轮收尾：集成、封版、复盘
depends_on: [{task: plan/r1, kind: children}]
machines: [<集成用的节点>]
---
```

**stack.json**（baseline 写 v0，close 写 vN）：

```json
{"version": "v1", "sealed": true, "round": 1,
 "infera": {"commit": "<sha>", "kit": "examples/<kit>", "params": {"...": "..."}},
 "image": "<tag@digest>", "patches": ["<patch>", "..."], "engine": "<名@版本>", "router": "<版本>",
 "client": {"agentx": "<IX commit>", "aiperf": "<版本>"}, "curve": "results/ 下 series v1 的点"}
```

`sealed: false` 表示本轮集成没有通过（或没有接受任何改动），版本不变，`version` 写当前版本。

**calibration.json**（可选：baseline 想给实验迭代筛选时写，plan 和实验读；不进判定，噪声底由台账里的正式复测算）：

```json
{"noise": {"same_node_pct": 0.01, "cross_node_pct": 0.5},
 "fast_vs_full": {"conc": 64, "ratio": 0.85, "rank_consistent": true},
 "dstep": {"ms_per_step_to_delta_pct": -1.6},
 "levels": {"L2": "dstep 固定 batch：约 20 min/变体", "L3": "AgentX fast 单点：约 30 min", "L4": "AgentX 正式单点：1.1–2 h"}}
```

## 问卷

- 第 1 轮：`questionnaire_r1.md`（22 行）。intake/r1 先预调研，把"建议"列填成具体值，写成工作区的 `questions.md`，然后结束这一轮等我填。
- 之后各轮：`questionnaire_rN.md`（增量问卷，约 8 行），由上一轮 close 的建议预填。

## 验收检查

intake、survey、baseline、plan、close 的字段检查是 `"$AGENT_SYS_PYTHON" -m agent_sys_v2.perfcheck <类型> "$AGENT_SYS_TASK"`，逐项打印 ok / FAIL。baseline 查的是"这套测量判得了 A/B"：工作点同一节点的正式复测次数、由此算出的复测噪声低于接受线、v0 的全量 GSM8K；不满足时不要硬凑，把状态改为阻塞交给我。
