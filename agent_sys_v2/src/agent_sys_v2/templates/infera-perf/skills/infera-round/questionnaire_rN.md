# 第 N 轮增量问卷（intake/rN 的 questions.md 模板，N ≥ 2）

用法（intake/rN）：读上一轮 close 的 report.md（"下一轮建议"一节）、`stack.json` 和台账，把下表的"建议"列填成具体值和理由，写成工作区的 `questions.md`，结束这一轮等我填。没填的行按建议处理。目标、部署和规则没有变化时不再问，沿用 `rules/project.md`。

| # | 问题 | 建议 | 你的意见 |
|---|---|---|---|
| 1 | 上一轮的成果：v(N−1) 是否发版、做成什么（只要配方和报告 / Infera kit 与 patch / Infera PR / 发布镜像 / 上游 PR） | 按 close 的建议；默认只出 kit、patch 和报告 | |
| 2 | 是否重新做对标（survey/rN） | 是：外部局面每周都在变 | |
| 3 | 是否重新 profile（profile/rN） | 满足任一条就建议重做：工作点累计提升 > 10%、换了工作点、换了拓扑或引擎版本、候选已用尽 | |
| 4 | 是否重做基线（baseline/rN） | 否：close 的集成曲线 vN−1 就是本轮基线；换集群、客户端或引擎版本时才重做 | |
| 5 | 本轮候选的方向和排序（上一轮剩下的、对标差异、你的想法） | 按 close 的建议列出前 5 项 | |
| 6 | 目标、工作点或成功标准是否调整 | 不调整 | |
| 7 | 本轮预算：天数、节点·小时、每个任务的美元上限；可用机器有没有变化 | 沿用上一轮 | |
| 8 | 是否结束这个项目 | 否 | |

## 拿到回答后

- 按回答起草本轮任务（草稿，不要开始执行）：`survey/rN`（问 2 = 是）、`profile/rN`（问 3 = 是）、`baseline/rN`（问 4 = 是）、`plan/rN`（总是），以及问 1 选了交付时的 `develop/…` → `review/…`。依赖：plan/rN 依赖本轮的 profile 和 survey（没有就依赖 intake/rN）；要上机的任务在 frontmatter 写 machines。
- 更新 `project.yaml` 的 `round: N`；有变化时更新 goal、machines、`ledger.working_point` 和 `rules/project.md` 的对应行。
- 问 8 = 是：不起草本轮任务，report.md 写项目总结，并写明"项目结束"（验收检查据此不要求本轮的 plan）。
