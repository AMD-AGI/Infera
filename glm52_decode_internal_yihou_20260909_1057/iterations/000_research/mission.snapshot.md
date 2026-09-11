## Brief

参见goal

## Category

开发、测试、跑通

## Goal

参考/home/yihou/dev/git/SIKL/sikl/benchmarks/models/glm52/profile_decode.py的思路。在集成/shared_nfs/yihou/packups/glm52_mix_repro.packup_20260909-100729的优化的前提下，通过截取调用sglang内部形成一种快速设定关键配置，运行decode bench拿到结果的方法。

### Finish Standard

方法形成，并成功运行conc=16, ISL=70K, OSL=10K, MTP cache hit rate = 3.61的decode only的测试, 产出报告。（注意不是通过pd分离然后fake input, 而是要求截取sglang内部model forward并进行简单封装快速测试，不走调度）

## Reference material

1. /home/yihou/dev/git/SIKL/sikl/benchmarks/models/glm52/profile_decode.py
2. /shared_nfs/yihou/packups/glm52_mix_repro.packup_20260909-100729

## Rules

**MUST：在工作的任何时候严格遵循以下规则**

### General

1. 注意本集群上yihou book的机器中，有一台正在测试pd分离相关实验，不要抢占其机器（目前是crsuse2-m2m-036，后续可能变更）， 你暂时是crsuse2-m2m-055， 只允许查询并选择使用机器，不允许申请机器。
    1. for every machein we occupied on spur as you query, kill all the GPU heavy process except ours and supervisors
2. 每隔10分钟注入一次任务书”mission.md”，确保mission不在上下文
3. 使用agent team进行工作
    1. leader应建立20min轮询机制，确保teammate再正确路径上、并不缺少信息。
    2. 轮询首次发现问题仅仅记录，等下次轮询teammate没有解决再介入讨论分析。
4. 严格遵循用户级别CLAUDE.md的工作规范，如有调整，请交互式询问用户。
5. 先research、gather information、analysis. 再plan， 再创建子workspace(保证所有的临时实验活动都在其中进行)，再创建CLAUDE.md(备份原先的)，再开始工作。
6. 一般使用docker container执行，不直接操作host。
7. 借助LSP和serena工作。
8. 不允许删除不包含yihou子串的任何文件。
9. Use English when work, Chinese only for report to user.

### Task related

1. cuda graph一般耗时比较长，可以等30min左右，同时如需观测可以检测build目录的变化。
2. 调用现有的debug skill等

### Local related
