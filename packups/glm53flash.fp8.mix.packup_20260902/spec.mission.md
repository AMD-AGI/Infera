## Brief

infera是一个专注分布式一键部署的大模型部署库，支持sglang、vllm、atom等engine.

glm5.3系列是Glm最新的模型。任务目标为将Glm5.3系列的各种基本支持添加到infera

## Category

experiment & integration & update

## Goal

具体为跑通infera + sglang + mix/mooncake 1p1d

任务可能几大范畴：

1. 支持glm5.3簇
2. 如需要：更新infera sglang镜像中sglang base, 增加需要的patch
3. 添加e2e test测试，并设为默认ci disable
4. 添加examples/（flash（原+mxfp4）和大(原[pd/mix] + mxpf4[pd/mix])的都要）

具体可以分为以下几个阶段，多阶段可以使用TP4 + agent team并行展开，以提高效率：

1. 收集InferenceX等权威数据amd mi355对于glm5.3的公开性能数据
2. 跑通glm5.3 flash 官方版本 pd-mix
3. 跑通glm5.3  官方版本 pd-mix
    1. 需要进行fixlen压测：性能需大致与glm5.2（本地packup/网络公开数据）对齐。
    2. 需要进行optimus-agenticsuite caseA fix测试
    3. 性能需大致和glm5.2对齐，进入合理区间。
4. 跑通glm5.3 flash mxfp4版本 pd-mix
5. 跑通glm5.3 mxfp4版本 pd-mix
6. 跑通glm5.3 mxfp4 pd分离

优先完成glm5.3 flash mxfp4版本。

## reference material

1. 四个模型路径/apps/data/models 
2. glm5.2 deploy example: examples/sglang_1p1d_glm5.2
3. glm5.2 mix 测试packup:
    1. /home/yihou/dev/git.16-19/infera.glm52.mix.experiment/*mix*
4. glm5.2 1p1d 测试packup:
    1. ~/dev/git/infera.glm52.view/glm52.example.packup_20260804
    2. ~/dev/git/infera.glm52.view/par8.armB.dpaoff.kvaware.spur.packup_20260804
    3. 禁止参考该目录下(~/dev/git/infera.glm52.view/)其他packup, 可能为错误信息
5. glm 5.3 flash mix 测试packup:
    1. /apps/yihou/packups/glm53flash.mix.packup_20260830
6. mxpf4版本的原模型转换厂商运行方法说明
    1. https://huggingface.co/OneNexus/GLM-5.3-MXFP4
    2. https://huggingface.co/OneNexus/GLM-5.3-Flash-MXFP4

## rules

1. 使用agent team进行工作
    1. leader应建立10min轮询机制，确保teammate再正确路径上、并不缺少信息。
    2. 轮询首次发现问题仅仅记录，等下次轮询teammate没有解决再介入讨论分析。
2. 本机为mi355 8卡环节，在本机进行，并行调试任务可以TP4+TP4同步进行提高效率，PD可以尝试单个node TP4 1p1d
3. cuda graph一般耗时比较长，可以等30min左右，同时如需观测可以检测build目录的变化。
4. 首先参考/apps/yihou/packups/glm53flash.mix.packup_20260830/Dockerfile.sglang.glm53，修改deploy/docker/Dockerfile.sglang使用最新的sglang镜像：lmsysorg/sglang:v0.5.18-rocm720-mi35x。 所有实验基于此dockerfile build出的image进行调试和添加patch
    1. 如果修改较少，直接进行实验
    2. 如果除了base之外有其他较大修改，可能影响其他模型，请和用户交互。
5. 严格遵循用户级别CLAUDE.md的工作规范，如有调整，请交互式询问用户。
6. 先research、gather information、analysis. 再plan， 再创建子workspace(保证所有的临时实验活动都在其中进行)，再创建CLAUDE.md(备份原先的)，再开始工作。
7. 一般使用docker container执行，不直接操作host。
8. 尽量不要更改host（包括根目录文件系统， 系统状态）， 如需更改，询问用户。
9. Use English when work, Chinese only for report to user.

### debug strategy

1. 调用现有的debug/debug iteration skill等
2. 遇到bug先research sglang issue/pr有没有相关同样问题和修复。其余仿照debug iteration skill发方法
3. rmda调试时首先参考tool rdma prefight/mode等组件调研结果（glm5.2 example里有），并且在每次e2e高成本实验前，请先用快速mvp小脚本验证猜想
4. debug时单次e2e实验尽量进行独立多方面猜想，并独立验证。从而缩短debug时间周期。
