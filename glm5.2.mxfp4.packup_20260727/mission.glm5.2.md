## Brief

在 infera库中使用sglang engine运行glm5.2 mxfp4版本.

## Goal

如非用户声明，一切实验以sglang为engine.

1. 单node mix运行glm5.2 mxfp4版本，得到正确输出，压测conc=64通过
2. 双node pd（mooncake）分离运行glm5.2 mxfp4版本，得到正确输出， 压测conc=64通过。
3. 双node pd（moriio）分离运行glm5.2 mxfp4版本，得到正确输出， 压测conc=64通过。

## reference material

1. test目录下Glm5.1 fp8版本的成功运行方法，包括vllm 单机/pd, sglang单机，atom单机。
2. /mnt/vast/xiaobo/models/GLM-5.2-MXFP4：模型位置
3. /home/yihou/dev/git.16-19/infera.vllm.image.update/glm_sglang_mix.packup_20260723: 完整的glm5.1 fp8版本sglang本地集群运行经验
4. /home/yihou/dev/git.16-19/infera.vllm.image.update/moriio_pd_fix.packup_20260723: 完整的glm5.1 fp8版本vllm + morrio的本地集群运行经验。
5. /home/yihou/dev/git.16-19/legacy.infera/infera.fuck/pd_1p1d_dpa_8k1k_20260714_235121： deepseek v4可以借鉴的sglang mooncake pd分离的本集群经验
6. examples/deepseek_v4/engine/pd_mooncake/sglang：有一些标准化的sglang mooncake pd分离经验，但需结合第五点适配本地集群。

## rules

1. cuda graph一般耗时比较长，可以等30min左右，同时如需观测可以检测build目录的变化。
2. first, gather & analysis information, then plan, then update CLAUDE.md to set this as the main goal. then execute.
3. use english when working to avoid punc issue of tool call.

### debug strategy

1. 调用现有的debug skill等
2. TODO: 方法论，技巧

## mechines & playground

1. 参考debug loop相关skill/plugin
2. 参考cluster rdma的相关skill/plugin
