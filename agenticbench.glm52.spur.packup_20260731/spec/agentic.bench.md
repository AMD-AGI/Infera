# Agentic bench verify

## Brief

根据infera对glm的运行方法（由实验给出）和agenticbench repo. 正确运行agentic bench在infera + glm5.2的测试。并拿到结果。

由于dpa + pd + mtp的成果我们尚未与aware合并，不要打开mtp

## Goal

1. 基于infera glm 最新的glm5 + pd + dpa + mtp的生产及方法。 使用agentic bench的方法进行bench， 收集并汇报以下数据：
    1. 简单的正确性验证。
    2. 经典指标
    3. mtp接受率
    4. session并行数，session turns, context length (如果由p50 p90 p99等分位数据更好。)

## Material

1. 最新的代码版本和相关修复&最新的运行方法: 
    1. 正确的kvd/kvaware的官方pr分支：/home/yihou/dev/git/infera.kv.fix
    2. 正确的kvd/kvaware在其他集群的运行方法：glm52.mxfp4.spur.mooncake.packup_20260731_final_deliverable
    3. 本地集群的使用方法见skill
    4. 本地集群包含mtp的运行方法参见 glm52.mxfp4.spur.mooncake.packup_20260731_main_converged，但是不用打开mtp，所以不用相关patch.
2. agentic bench的代码: /home/yihou/dev/git/Optimus-AgenticBench/
3. agentic bench我们需要的case的的使用说明: /home/yihou/dev/git/Optimus-AgenticBench
4. mooncake event error patch: https://github.com/AMD-AGI/Infera/pull/56/changes#diff-c72fb30b9f9f89d8d29e8727a86227697b91da79f19f99acc5b0f60b36c2ca67

## Rules

1. 严格遵循用户级别CLAUDE.md的工作规范，如有调整，请交互式询问用户。
2. 如有实验/debug部分，可以并行展开（e2e实验比较慢，此举可以提升速度。），并且尽量进行多个猜想, 实验进行多个独立互不干扰猜想的同时验证（比如多个print的代码注入等）。
3. 为了快速实验，先打kv aware的镜像，然后需要的代码修改直接在现有镜像里完成，不要打额外的patch。
