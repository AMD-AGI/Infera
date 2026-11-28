# GLM-5.2 P4D4（Infera + vLLM）aus 集群：阶段性总结（截至 11-28 01:40）

计划见 `PLAN.zh-CN.md`，执行过程与问题见 `LOG.zh-CN.md`，复现见 `kit/README.zh-CN.md`。所有成绩都是单次运行，C152、MTP 草稿 4（合成接受 3.33）、3600 s。

## 1. 结果

| 阶段 | 配置 | 节点 | tok/s/GPU | 交互 p50 / p90 | ITL p50 / p90（ms） | TTFT 均值（s） | GSM8K（C152） | AL / golden 3.33 |
|---|---|---|---|---|---|---|---|---|
| 参考 | crusoe a49（v0.34 + 覆盖层） | crusoe v3-017 | 71,061 | 72.2 / 61.5 | 13.9 / 16.2 | 8.18 | — | — |
| B | a49 原样复现 | n04-29 | 69,682（−1.9%） | 71.5 / 59.4 | 14.0 / 16.8 | 8.40 | 0.9689 | — |
| C | C1：nightly de7cf7f8 + 12 个上游 / fork PR | n04-25 | **70,672** | 70.1 / 59.6 | 14.3 / 16.8 | 8.36 | 0.9682 | 3.254（pass） |
| D | C1 + THP-off + decode GEMM 调优表 | n04-29 | 69,744 | 69.5 / 59.6 | 14.4 / 16.8 | 8.55 | 0.9727 | 3.277（pass） |
| E | D 去掉 GEMM 调优表 | n04-29 | 70,455 | 69.3 / 59.6 | 14.4 / 16.8 | 7.84 | （同 C1） | |
| F-B0 | C1，路由选 decode rank（请求权重 0） | n04-25 | 69,998 | 68.5 / 59.4 | 14.6 / 16.8 | 8.25 | （同 C1） | |

客户端：B 用 recipe 原来的 InferenceX 918524ff（与 crusoe 相同），C–F 用 InferenceX main d490919c（agentx-harness 1.0.14）；吞吐口径两者相同。

## 2. 结论

1. **复现对齐**：a49 在 aus 上 69,682，比 crusoe 低 1.9%，在 3% 的标准内；各段延迟接近。aus 节点是 amdgpu 6.19.14（有 peer memory），
   实测 dma-buf 和 peer memory 注册都会把显存记两次账，用 GEM（与预期一致）；单节点 PD 走 RDMA 回环（`MC_DISABLE_HIP=1`）。
2. **只用上游 / fork PR 的 nightly（C1）达到 crusoe 最好配置的水平**：70,672，GSM8K 和 AL 门禁都通过（MTP 草稿层保持 FP8，
   fork #12 生效）。12 个补丁都是对应 PR 的原样 diff：别人的 open PR 4 个、补充 / rebase 的 fork PR 4 个（#11–#14）、新开的 fork PR 3 个（#15–#17）、已有的 fork PR 1 个（#7）。
3. **不能上游的项没有带来收益**：同节点 D（带 GEMM 调优表）69,744 对 E（不带）70,455，ITL 相同 → 去掉 GEMM 调优表不损失。
   THP-off 的作用还没测到（同节点的 C1 那一轮被抢占）。
4. **decode rank 改由路由选**：F-B0（只按预约的输入 token，请求权重 0）比同节点的 C1（vLLM 选，KV 权重 14）低 1.0%，ITL p50 高 0.3 ms；
   路由把请求分得很均匀，但各 rank 的生成 token 数最多 / 平均 1.039（C1 是 1.015）：只按输入 token 记账，看不到输出长度的差别。F-B1（计入流式输出）那一轮被抢占，还没有数。

## 3. 正在 / 将要做（37014 被抢占，预计 09:07 重新开始）

| 项 | 目的 |
|---|---|
| F-B1（W=0） | 路由计入输出增长能否追上引擎选 |
| C1 重跑（带每 10 s 的引擎指标采样） | C1 的重复性、引擎选 rank 时各 rank 的均衡度 |
| F-B1 W=125,000 | 加请求数权重（约一个平均输入长度） |
| 同节点 C1 对 E | THP-off 的作用 |

## 4. 过程中修掉的问题（细节见 LOG）

- nightly 镜像：补丁路径；#17 在模块顶层用了 `@gluon.jit`（无 GPU 时 import 失败，已在 PR 里修）。
- kit：引擎命令多行导致 `bash -c` 只执行第一行（PP1、无 MTP、OOM）；AgentX / AL 门禁的 ZMQ ipc 路径过长；InferenceX 汇总需要
  `KV_OFFLOAD_BACKEND_METADATA`、`IS_MULTINODE=true`；AL 门禁要 `/metrics` 地址。
- 测试：C0 测试 checkout 只带了 `vllm/` 改动，回归测试 46 个"失败"是测试文件版本不对；打上各 PR 自带的测试改动后全部通过。#15 的测试伪造对象缺字段（已修）。
- 运行：`git pull --rebase` 替换了正在被 bash 读的脚本（Stale file handle）→ 每个任务从节点本地快照运行；节点检查和部署撞车 → GPU 测试只在节点空闲时排。
