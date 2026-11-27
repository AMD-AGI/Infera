# 执行记录（新的在上）

## 11-27

- 17:52 37014 开始，节点 smci355-ccs-aus-n04-25 / n04-29，两台都是 amdgpu 6.19.14，都有 `ib_peer_mem`；GPU i 和 ionic_i 在同一个 PCIe 交换芯片下（与 crusoe 相同），8 块网卡都是 ACTIVE。
- 17:55 问题：节点检查里三个显存注册测试都没跑起来（`docker run --group-add render` 用了组名，镜像里没有 render 组），GEM=1 是失败后的默认值。改成传主机的数字 gid（`test_C1.sh` 也一样），用 `srun --overlap` 在两个节点上重跑节点检查，和正在进行的镜像构建并行。
- 15:34 静态准备进展：
  - 节点检查（`probe/probe_node.sh`）：在 nightly 原版镜像里跑 150 GiB 的 dma-buf / peer memory / GEM 三个用例，镜像自带和主机的 `libionic` 各一轮，按结果自动定 GEM 和 libionic。
  - 阶段 B 回放（`replay/`）：crusoe 脚本原样组装，路径替换用 sed，逻辑改动在 `aus.patch`（9 处）。
  - InferenceX main 版的 eval / AgentX（`kit/`）。
  - 队列：每个节点各构建一次 v0.34.0 镜像，然后由空闲节点领走阶段 B 的 fast 和 full。
  - 核对了在线 FP8 的范围：模型 78 个主干层，NextN 草稿层编号 78，FP8 只作用于 0–77，符合规则。
  - nightly 补丁集、路由侧选 rank 由子 agent 并行准备。
  - 一个小问题：一次提交带进了子 agent 还在生成中的 `patches/meta/` 草稿，等补丁集定下来后整理。
- 15:10 工作目录挪进 Infera 分支 `dev/pd_opt/vllm_glm52_agentx` 的 `llying/vllm-p4d4-aus-20261127/`；原路径 `bench_agentx/vllm-p4d4-aus-20261127` 是指向这里的软链接（37019 的批处理脚本写的是原路径）。
- 14:55 提交 37019（2 节点、24 h、`afterany:37014`），因为 QOS 单作业最长 24 h；用户在问为什么又提交了一个作业，取不取消由用户决定。
- 14:00 核对测试方法：吞吐的计法与 InferenceX 一致（a49 复算 ≈ 71,07x）；eval 不一致（应该是全量 GSM8K 1,319 题、并发等于计分并发、阈值 0.90，外加 AL 门禁），客户端 918524ff 比 main 落后 3 个月。详见 PLAN 第 9 节。
- 13:50 集群信息：aus 节点混有 amdgpu 6.16.13 / 6.19.14；网卡和 crusoe 一样是 ionic_0..7、GID index 1；模型在 `/perf_apps/data/models/GLM-5.2-MXFP4`；docker 以 root 运行，读不了 /home（NFS root-squash），所有挂载路径都放在 /perf_apps 下；节点上没有 /scratch。
