# glm5.2 example create

## Brief

infera已经提供了glm5.2生产及部署姿势并进行了一些agentic的bench, 包括使用Optimus-agenticbench和客户提供了自身的bench方法。现需对可与交付正式部署的example。

## Goal

分析各成功运行的示例，仿照examples中其他样例格式，给出infera部署glm5.2 sglang + mooncake 1p1d + dpa + mtp的正确方式，要求为一键/几键脚本。

要求：

1. example中绝对不包含以下信息，如有必要提醒避免踩坑，写入注释和readme的对应章节：
    1. 本地信息
    2. 过程信息
    3. 不记录adhoc或者实验中制作docker image的方法，替换为标准发布镜像，先记录为infera-sglang-0.2.0占位符，待随后确定。
2. bench client启动脚本不提供。只给出简单服务检验方法和使用sglang自身的server bench对服务器的测试以供参考
3. 增加dp和no-dp的选项。
4. 提供result文件夹，分别记录conc=8下infera自身bench和客户bench方法二者的结果和分析。
5. 给出dpa和nodpa的部署方式，并且在readme中指出建议配置。
6. 提供readme中rdma prefight（判断有几个网卡，有没有bdf, 有没有peermem，从而给出部署flag变化的）的使用方式，提醒和建议。
7. 部署脚本将本地化的flag提出真实脚本，附加wrapper脚本，并告诉用户，如果需要根据集群修改，改这里。

## reference material

1. infera分支：main
2. infera 已有 example 样本：examples/
3. vultr集群跑MAD的agentic测试成功packup: ../infera.merge.liying.kv.mtp/agentx.caseA.customer.packup_20260803
4. sglang自身server bench fixlen成功packup: ../infera.merge.liying.kv.mtp/fixlen.glm52.fullfeature.packup_20260801
5. vultr集群infera自身agentic bench的packup（包含服务部署和测试方法）: ../infera.merge.liying.kv.mtp/par8.glm52.dpaoff.packup_20260803
6. CruSoe(spur)集群infera自身agentic bench的packup（包含服务部署和测试方法）: ../infera.merge.liying.kv.mtp/agenticbench.mtp.caseA.packup_20260801
7. CruSoe(spur)集群MAD的agentic测试成功packup: 暂时没有，待用户提供。
8. 客户bench 方法: https://github.com/ROCm/MAD/pull/173/changes
9. 客户代码的本地repo: /home/yihou/dev/git/MAD

## rules

1. 严格遵循用户级别CLAUDE.md的工作规范，如有调整，请交互式询问用户。
2. 先research、gather information、analysis. 再plan， 再创建子workspace(保证所有的临时实验活动都在其中进行)，再创建CLAUDE.md(备份原先的)，再开始工作。
3. 如有实验/debug部分，可以并行展开（e2e实验比较慢，此举可以提升速度。），并且尽量进行多个猜想, 实验进行多个独立互不干扰猜想的同时验证（比如多个print的代码注入等）。
4. 本目录下有很多之前实验的packup，如非用户允许，不允许参考用户在Materials中指定的以外的packup.
5. Use English when work, Chinese only for report to user.
