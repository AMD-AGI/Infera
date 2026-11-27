# 执行记录（新的在上）

## 11-27

- 15:10 工作目录挪进 Infera 分支 `dev/pd_opt/vllm_glm52_agentx` 的 `llying/vllm-p4d4-aus-20261127/`；原路径 `bench_agentx/vllm-p4d4-aus-20261127` 是指向这里的软链接（37019 的批处理脚本写的是原路径）。
- 14:55 提交 37019（2 节点、24 h、`afterany:37014`），因为 QOS 单作业最长 24 h；用户在问为什么又提交了一个作业，取不取消由用户决定。
- 14:00 核对测试方法：吞吐的计法与 InferenceX 一致（a49 复算 ≈ 71,07x）；eval 不一致（应该是全量 GSM8K 1,319 题、并发等于计分并发、阈值 0.90，外加 AL 门禁），客户端 918524ff 比 main 落后 3 个月。详见 PLAN 第 9 节。
- 13:50 集群信息：aus 节点混有 amdgpu 6.16.13 / 6.19.14；网卡和 crusoe 一样是 ionic_0..7、GID index 1；模型在 `/perf_apps/data/models/GLM-5.2-MXFP4`；docker 以 root 运行，读不了 /home（NFS root-squash），所有挂载路径都放在 /perf_apps 下；节点上没有 /scratch。
