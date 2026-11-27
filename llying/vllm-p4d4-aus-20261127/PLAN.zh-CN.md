# GLM-5.2 P4D4（Infera + vLLM）在 aus 集群复现、升级 nightly、上游化：计划（2026-11-27）

参考：`AMD-AGI/Infera` 分支 `dev/pd_opt/vllm_glm52_agentx` 的 `llying/`（本地克隆 `../infera-vllm-glm52-ref`）。
风格参考 Infera #210：一个 example 目录（`env.sh` / `up.sh` / `smoke.sh` / `eval.sh` / `down.sh` + README 表格），补丁旁边配一份状态表。

## 0. 对齐对象（crusoe 上的结果，都是单次）

| 点 | 配置 | tok/s/GPU | 说明 |
|---|---|---|---|
| a49 B（吞吐最高） | 草稿 4 + C152 + 全部覆盖层 + dense 在线 FP8 + MXFP4 indexer | 71,061 | 接近饱和（TTFT 均值 8.2 s），计分窗口 9 次撤队列 59 s |
| a53 B | a49 + decode 显存 0.97 | 71,314 | 与 a49 的差在噪声内 |
| a44 B（门禁来源） | 同 a49 但 C104 | 60,170 | a49 本身没跑门禁，正确性证据来自 a44 |

a49 用到的所有改动（共 18 项）见第 6 节的清单。

## 1. 阶段 A：节点体检与驱动决策（拿到节点后约 30 min）

脚本 `probe_node.sh`，每个节点跑一次，结果写进 `results/probe-<node>/`：

- amdgpu 版本（`modinfo amdgpu` / `/sys/module/amdgpu/version`）。集群里 6.16.13 与 6.19.14 两种都有（`kfd619/` 11-13 的记录：n04-29 是 6.19、n04-33 是 6.16）。
- peer memory：`lsmod | grep -E 'amdgpu_peer_mem|ib_peer_mem'`，`/sys/kernel/mm/memory_peers/`。
- 网卡：rail 个数与名字（crusoe 上是 8 块 ionic），GID index，拓扑（为 `MC_CUSTOM_TOPO_JSON` 准备）。
- 显存双倍计数测试：复用 `kfd619` 的 `repro-gemfd` 方法，注册前后看 `headroom` 与真实可用显存。
- 本节点 RDMA 回环带宽（GPU 显存 → 本节点另一块 GPU，走网卡）。

决策表（实测为准）：

| 驱动 | peer memory | 显存注册方式 | GEM 分配器 |
|---|---|---|---|
| 6.16 | 有 | peer memory（`MOONCAKE_DISABLE_HIP_DMABUF=1`） | 不用（`P_GEM=0 D_GEM=0`） |
| 6.16 | 无 | dma-buf，有双倍计数 | 用（与 crusoe 相同） |
| 6.19 | 有或无 | 双倍计数问题都在 | 用 |

单节点内 PD 也必须走 RDMA：`MC_DISABLE_HIP=1`（否则 Mooncake 优先选 HIP），作业结束时用网卡计数（`nic-delta.txt`）确认 8 块网卡都有流量、tx ≈ rx。

## 2. 阶段 B：在 aus 上原样复现 a49（v0.34.0 镜像）

- 在计算节点上构建 `llying/vllm-image-v0340-20261117/Dockerfile`（登录节点不做重活），镜像名与 crusoe 相同。
- 部署脚本沿用 `vllm-pd-opt-20261123/deploy/` 与 `once/p4d4-pair12.sh`，只改集群相关的部分：模型路径 `/perf_apps/data/models/GLM-5.2-MXFP4`、网卡名与拓扑、GEM 开关、Slurm 参数、排除节点列表。
- 顺序：
  1. 门禁（GSM8K 200、needle 64 × 16k、真实接受长度）。
  2. C104 计分，与 a44 的 60,170 对比。
  3. C152 计分，与 a49 的 71,061 对比。
- 两个节点各起一套 P4D4（每节点 8 卡，一套就占满），两边同时跑，得到同一配置的两次结果，也能看出节点间差异。
- 对齐标准：差在 3% 以内算对齐；超过 3% 就按 decode 步长、KV 拉取、prefill 忙碌率逐段对比 crusoe 的数据（`results/a44`、`results/a49`）找原因。

## 3. 阶段 C：nightly 镜像 + vLLM main + 补丁

- 基础镜像固定为 `vllm/vllm-openai-rocm:nightly-rocm100-de7cf7f8cb5dc47d2e65207e0fabbe2c8f0d73b3`（11-27 06:06 构建，ROCm 10.0）。补丁全部对这个提交做；以后要换 nightly 时整套重新验证一遍。
- 镜像里的 Mooncake 换成 main（包含 MC#4540、MC#5680）+ 未合入的 MC#4541；GEM / VMM 部分用 MC#5584 / MC#5587 的原样 diff。
- 每个补丁都是对应 PR head 的原样 `git diff`，目录 `patches/vllm/<PR号>-<短名>.diff`，旁边的 `PATCHES.md` 记录：PR、head sha、状态、作者、为什么需要、不打会怎样。
- 已在 main 的不打补丁：#67656、#68614、#66799、#68006（U13 的一半）、#51392（在线 FP8 的机制）。
  其中 #68614 在 a32 测过 −2.8%（结果混着 prefill 停顿），#66799 默认开，这两项在 nightly 上要单独 A/B 确认没有负收益。
- 补丁的分类与处理（PR 状态是 11-27 的）：

| 类 | 项 | 处理 |
|---|---|---|
| 别人的 open PR，原样能满足目标 | #69540、#68845（CPU 层）；#68596（U12 的等价实现，要先确认与我们的 port 等价） | 原样 diff；不满足时在原 PR 下留评论，或开补充 draft PR |
| 别人的 open PR，要改才满足目标 | #61985（MXFP4 indexer：只能作用于目标模型层，草稿层要保持 FP8）；#58710（sync skip，我们用的是自定义等价实现）；#60909（U26 精确锁页） | 能用原 PR 就用原 PR；不能用就开补充 draft PR，补丁 = 我们的 PR diff |
| 我们已开的 PR | #69616（U17）、#69617（U19）、fork #7（KV 感知 DP 均衡，还只在 fork）、Infera #246 / #248 / #250 / #251、MC#4541 | 先 rebase 到 main；fork #7 要转成 vllm-project/vllm 的 draft PR |
| 还没开 PR | Triton decode top-k（U34）、FP4 dispatch + 一次 AG 打包（U33）、U13 在 main 上重做、U12（如果 #68596 不行） | 开 draft PR，补丁 = PR diff |
| 看起来不能上游 | 见第 4 节 | 补丁单独放在 `patches/local/` |

- 验证：每个补丁先过单元测试（沿用 `scripts/test_*.py`，改成对 nightly 文件）；整套跑门禁，再跑 C104 和 C152。

## 4. 阶段 D：带"不能上游"项的完整版本（与最佳性能对齐）

我初步认为不能上游、或者不该算作 vLLM 补丁的项（请确认）：

| 项 | 原因 |
|---|---|
| THP off（`.pth` 在解释器启动时调用 prctl） | 是主机 / 进程层面的规避手段，不是 vLLM 功能；可以改成容器启动参数或环境变量，不进补丁 |
| decode BF16 GEMM 调优表（U35） | 属于 aiter 的 `model_configs`，可以提给 aiter，不是 vLLM 补丁；作为 aiter 的数据文件挂载 |
| GEM / VMM 的 `LD_PRELOAD` 分配器 | 是驱动 bug 的规避手段，依赖节点类型；只在 6.19 等节点上开 |
| `VPD_DP_LB_KV_CAP_TOKENS` 这类手填容量 | 上游版本应该从引擎自己读出 KV 容量 |
| 测量用的钩子（steplog、prefill usage 日志） | 只在诊断时开 |
| MXFP4 indexer、dense 在线 FP8（会改数值） | 规则上允许，但要上游合入并通过 eval；先算作"可上游"，放单独开关 |

在阶段 C 的镜像上打上第 3、4 节的全部补丁，跑 C152（和 C104），目标是与阶段 B 的结果差在 3% 以内。

## 5. 阶段 E：去掉不能上游的项

在阶段 D 的基础上，逐项关掉第 4 节确认不能上游的项，跑 C104 和 C152，并扫描 C136 / C152 / C168 找到新的最好点。
每项的贡献按 crusoe 的成对数据估计（THP-off、GEMM 表等）；差值小于 0.5% 的当作噪声。

## 6. 阶段 F：把 decode DP rank 的选择放到 Infera 路由

现状（`DP-RANK-SELECTION.zh-CN.md`）：Infera #250 让路由不指定 rank，fork #7 让 vLLM 按"等待 + 在跑 + 14 × KV 用量"选 rank；单次测得合计约 +2.3%。
目标：与 Infera + sglang 一致，由路由选 rank。

1. **B0**：直接用路由现有的 `--decode-input-demand on` + `--decode-request-weight W`（#210 的做法），vLLM decode 按 rank 展开注册，用 `X-data-parallel-rank` 指定 rank。
   先排查 m13 那次 decode 崩溃；连带修好 KV 事件端口（Infera #196 / U7）。
2. **B1**：路由记账补上输出增长（decode 流每 N 个 token 追加一次）和请求结束时释放；参考 Dynamo 的做法。
3. **B2（可选）**：decode worker 定期上报每个 rank 的真实 KV 用量，路由用它修正记账。
- 在同一镜像上成对比较：A（#250 + fork #7）对 B0 / B1 / B2，C152 和 C104 两个点。
- 每次都记录：最忙 rank / 平均请求数、上下文不均（`scripts/dp_imbalance.py`）、rank KV > 95% 的时间占比、吞吐、ITL。
- 如果 B 的结果比 A 差，就按这些指标定位原因：rank 定得太早（prefill 之前就定了）、没算输出增长、权重不合适。

## 7. 交付物（照 #210 的风格）

- `examples/vllm_p4d4_glm5.2/`：`env.sh`（站点相关：驱动 / GEM、网卡、模型路径）、`engine/up.sh`、`smoke.sh`、`eval.sh`（门禁 + AgentX）、`down.sh`、README（每个阶段的结果表与对应命令）。
- `deploy/docker/Dockerfile.vllm-nightly` 与 `patches/`（`PATCHES.md` 状态表）。
- 每个 PR 一份说明（英文正文 + 验证数据），PR 正文里只写测过的数。

## 8. 时间与资源

- 每次计分 3600 s，加上启动、预热、门禁，约 1.5–2 h；两个节点并行，作业最长 24 h。
- 估计轮数：A 0.5 天；B 约 4 轮；C 包括构建、补丁和测试约 1–1.5 天，加约 4 轮；D 约 2 轮；E 约 4–6 轮；F 约 6 轮。
- 补丁整理和 PR 准备在等作业的时候做；对外操作（开 PR、评论）每次先找用户确认。

## 9. 11-27 用户答复后的修订（取代前文冲突处）

- 性能只对齐 C152（a49，71,061）；C104 不单独跑。
- 改数值的两项（dense 在线 FP8、#61985 只作用于目标层）：符合 InferenceX 规则、eval 通过、有提升就算在 recipe 里。
- `VPD_DP_LB_KV_CAP_TOKENS` 不是独立 feature，在 fork #7 的上游 PR 里改成从引擎读 KV 容量。
- 阶段 E 保留跑起来必需的集群绕过项（GEM、THP-off），只去掉性能类的不能上游项。
- PR 先开在 llying-001 的 fork 仓，自主执行，不等确认。
- full 计分约 6–8 轮（B 1、C 1、D 1、E 1–2、F 2–3）；调试和补丁验证用 fast 模式（`AIPERF_EXPERIMENTAL_FAST=1`：1200 s、每条通道 1 个预热，并发不变），只在对齐性能时跑 full（3600 s、每条通道 10 个预热）。

### 测试方法与 InferenceX 对齐（11-27 核对 InferenceX origin/main d490919c）

- 吞吐口径一致：(输入 + 输出 token) / (最晚结束 − 最早开始)，只算成功的计分请求，GPU 分母 = prefill + decode 共 8 卡。
  a49 用 `window_tput.txt` 复算得到 4.545 req/s × 125,090 token / 8 卡 ≈ 71,07x，与 71,061 一致。输入 token 占 99.2%（前缀命中也计入，这是 InferenceX 的定义），所以吞吐随并发上升。
  sglang 的 53,351 是 C104，口径不同：比较要按同并发，或者按 InferenceX 的方式在"吞吐 – 交互 p90"的 Pareto 曲线上比。
- 合成接受符合规则：草稿 4 对应 golden AL 3.33，vLLM 的 `rejection_sample_method: synthetic` 就是官方要求的写法；a49 实际 3.322。
- **eval 不一致，要改**。InferenceX 的做法：
  - lm-eval，全量 GSM8K 1,319 题，5-shot，temperature 0，开 chat template；
  - 并发 = 这条曲线最高的计分并发（这里是 152）；
  - 真实接受（不开合成接受）；
  - 阈值 0.90，strict-match 和 flexible-extract 都要过；
  - 外加只报告的 AL 门禁：SPEED-Bench coding 80 题，AL 要在 golden 的 0.95–1.05 内。

  我们之前的门禁（GSM8K 200 题、并发 64，needle 64 × 16k）只保留作快速自检。
  `eval.sh` 改成直接调用 InferenceX origin/main 的 `python3 -m infx.bench eval --concurrency <CONC>` + `infx.evals.validate_scores`，再加 `al-gate`。
- 客户端版本：recipe 固定在 918524ff（08-27，AIPerf 子模块）；main 已改用 agentx-harness 1.0.14（修过重复会话状态等问题）。
  阶段 B 先用 918524ff 对齐 crusoe，同时用 main 重算一次同一配置；之后各阶段都用 main 的 harness。
  元数据补上：prefill 写成 PP4 / TP1，decode 写成 DP4，并按 main 的要求设 `PREFILL/DECODE_GPUS_PER_WORKER`。
- 规则提醒：在线 FP8 的 ignore 正则要确认 NextN 层的编号就是 78（草稿必须保持发布精度）。

## 10. 两节点并行排程、GPU 利用率与提交（11-27 用户要求）

节点：37014（用户的 2 节点占位作业，预计 11-27 17:51 开始，24 h）；37019（2 节点，`afterany:37014`，后备）。QOS `batch` 单作业最长 24 h、每人最多 4 节点。
运行方式：每个节点一个队列执行器 `holder/holder.sh`（37014 用 `holder/attach.sh` 挂进去，37019 用 `holder/holder.sbatch`）。
队列：`holder/q-<节点>/`（指定节点）与 `holder/q-any/`（哪个节点空闲就跑）。

| 时间 | 节点 0 | 节点 1 |
|---|---|---|
| 0–2 h | 节点检查 → 构建 v0.34.0 镜像 | 节点检查 → 拉 nightly 原版镜像，先在 GPU 上跑原版 P4D4 的 fast（检查环境，并作为"无补丁 nightly"的数据点）；之后构建 nightly + 补丁镜像 |
| 2–9 h | B：fast 跑通 → 正式一轮（InferenceX eval + 计分） | C/D：nightly 补丁在 fast 模式下调通 |
| 9–12 h | D 正式一轮 | C 正式一轮 |
| 12–15 h | E 正式一轮 | E 补点 / F 的 fast 调试 |
| 15–21 h | F：B0 / B1 正式 | F：对照组 A 正式 |
| 21–24 h | 按 kit 的 README 从头复现 | 补测 |

- **GPU 利用率**：
  - 队列不留空，主任务之间用预先排好的补充测试填上（Pareto 曲线上的其他并发点、交换节点的重复轮）。
  - 镜像构建（纯 CPU）和 GPU 任务错开，不让两个节点同时只在构建。
- **节点差异**：开始时两个节点各用 fast 模式跑一次同一配置，看两个节点的差距。
  需要成对比较的（D 对 E、F 的 A 对 B），同一组的两边放在同一个节点上跑，或者交换节点各跑一次，不拿两个节点的结果直接相减。
- **提交**：工作目录在 Infera 分支 `dev/pd_opt/vllm_glm52_agentx` 的 `llying/vllm-p4d4-aus-20261127/`。
  - 每写完一个脚本、遇到一个问题、拿到一个结果就提交并推送（DCO 签名，推送前 `pull --rebase`，只改自己的目录）。
  - 原始日志和 trace 不进 git，只提交摘要。
  - 问题与处理记在 `LOG.zh-CN.md`。
- 镜像构建用的源码固定在 `0fcce33`（`../../infera-vllm-glm52-ref` 那份 clone），不随这个分支变化。
