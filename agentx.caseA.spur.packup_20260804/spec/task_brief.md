# External agentic bench

## Brief

infera之前使用自身的agentic bench套件对自身的glm5.2部署姿势做了一些agentic的bench, 现在客户提供了自身的bench方法。我们需要用此bench方法对服务进行测试，给出结果并分析。

https://github.com/ROCm/MAD/pull/173/changes

## Goal

在 yihou.dev.glm52.merged.experiment分支上，结合infera自身的部署经验。成功运行https://github.com/ROCm/MAD/pull/173/changes中的glm5.2的agentic bench，拿到结果并分析报告。

## reference material

1. vultr集群跑MAD的agentic测试成功packup: 随后我补给你
2. vultr集群infera自身agentic bench的packup（包含服务部署和测试方法）: par8.glm52.dpaoff.packup_20260803
3. CruSoe(spur)集群infera自身agentic bench的packup（包含服务部署和测试方法）: agenticbench.mtp.caseA.packup_20260801
4. 客户bench 方法和简单部署代码: https://github.com/ROCm/MAD/pull/173/changes
    1. 客户的部署方法如果不一样，请分析哪些地方值得学习，交由用户决策
5. 客户代码的本地repo: /home/yihou/dev/git/MAD

par8.glm52.dpaoff.packup_20260803的部署方法和workload, CruSoe仅供参考。

## rules

1. 严格遵循用户级别CLAUDE.md的工作规范，如有调整，请交互式询问用户。
2. 先research、gather information、analysis. 再plan， 再创建子workspace(保证所有的临时实验活动都在其中进行)，再创建CLAUDE.md(备份原先的)，再开始工作。
3. 开始工作前展示分析结果，包括需要pack哪些修复，还要额外做哪些修改，风险是什么，大不大。
4. 如有实验/debug部分，可以并行展开（e2e实验比较慢，此举可以提升速度。），并且尽量进行多个猜想, 实验进行多个独立互不干扰猜想的同时验证（比如多个print的代码注入等）。
5. 本目录下有很多之前实验的packup，如非用户允许，不允许参考用户在Materials中指定的以外的packup.
6. Use English when work, Chinese only for report to user.
