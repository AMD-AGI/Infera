## Brief

使用Optimus-AgenticBench对infera中glm5.2最完整特性支持的服务部署进行bench.

## Goal

1. Infera针对glm5.2 + kvaware + kvd + mtp + pd + dpa有一系列修复和完整的支持。
2. Infera有在kvaware + kvd + pd的场景下对glm5.2进行过简单的agenticbench。

综上，需要在1中最新的全量特性支持的基础上，运行2中caseA fix的测试，拿到结果。并按照xxx文档整理出报告。

### Task breakdown

1. 在开展bench之前，应首先验证1中各个特性已真实打开（如kvaware、kvd等）。
    1. 在bench中，应简单分析并设置如kvaware权重为合适的数值。如实验中在不影响测试性能的情况下有数据可以记录说明这些指标。
2. 首先使用sglang自有server bench, bench 各个ISL/OSL分位conc =(1,32,64,128)的情况，收集所有结果。成功结束后， **使用packup技能packup实验准确复现资料**。
3. 开始正式实验和调试，成功结束后， **使用packup技能packup实验准确复现资料**。。
4. 按照类似xxx的格式，给出每项数值的分析，最后结合各项数值，分析整个实验结果及其合理性。

## reference material

1. glm5.2 + kvaware + kvd + mtp + pd + dpa最终分支：/home/yihou/dev/git[vultr suffix: a16-19]/infera.merge.liying.kv.mtp (pwd)
2. glm5.2 + kvaware + kvd + mtp + pd + dpa最终vultr成功实验packup:
    1. 最新：liying_rest_pr56.packup_20260801
    2. 部分：glm52.merged_branch_image.packup_20260801 
3. kvaware + kvd → agentic bench 实验packup:
    1. /home/yihou/dev/git[vultr suffix: a16-19]/infera.glm5.2.experiment/agenticbench.glm52.spur.packup_20260731
    2. /home/yihou/dev/git[vultr suffix: a16-19]/infera.glm5.2.experiment/kvd.rocm.hostalloc.packup_20260731

### Case A original spec

仅供最后分析对照：

- Input Shape (P50, P90, P99): 74K, 155K, 235K tokens
- Output Shape (P50, P90, P99): 320, 3.3K, 17K tokens
- Cache Hit Rate: 88–90%
- Turns per Session (P50, P90, P99): 3, 20, 103
- Inter-Turn Delay (P50, P90, P99): 4s, 31s, 4m
- Speculative Decoding: 56% acceptance rate, 5 draft tokens
- SLA Target: P50 E2E < 4.5s

## Cluster Environment Notice

### Vultr only

1. infera服务镜像应已在之前的测试中有, 因此基于之前的镜像直接测试，如有少量patch请先inplace直接修改并在本地记录，方便随后整理成正式的。
2. agenticbench的bench verify参考部分配置是基于crusoe集群的特性：如无法使用ionic网卡，必须使用mlx5接受性能回退。应识别此类flag并予以摒弃。

### Crusoe Only

1. glm5.2 + kvaware + kvd + mtp + pd + dpa实验记录是在vultr集群的，适配crusoe集群可能需要参考最初bench资料调试。
2. 最新镜像本集群应无，因此在bench前，应先完成：
    1. build出最新镜像被被分在nfs上
    2. 结合本集群特性，调试成功运行glm5.2 + kvaware + kvd + mtp + pd + dpa基础实验。

## Rules

1. 严格遵循用户级别CLAUDE.md的工作规范，如有调整，请交互式询问用户。
2. 本文件夹和其他worktree内存在其他轮次的实验packup，信息陈旧有错误，因此用户提供以外的packup资料和试验资料禁止参考（除非用户声明）。
3. 如有实验/debug部分，可以并行展开（e2e实验比较慢，此举可以提升速度。），并且尽量进行多个猜想, 实验进行多个独立互不干扰猜想的同时验证（比如多个print的代码注入等）。
4. 正确Needle测试上不要耗费太多时间，如需验证该项，请单独直接打开chat template + 官方推荐temperature + top p解决。
