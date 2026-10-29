# 把 infera + sglang 的 GLM-5.2 AgentX Pareto 发布到 InferenceX：还差什么、怎么做、风险与工期（10-27）

对象：[Infera PR #210](https://github.com/AMD-AGI/Infera/pull/210) 的第一版 Pareto（MI355X，GLM-5.2-MXFP4，PD，PP4 prefill + DPA decode，7 个点）。
本报告只做分析，没有上机。InferenceX 的规则、流程和先例取自 `SemiAnalysisAI/InferenceX` main `7ca692b9`（10-27）、其固定的
`NVIDIA/srt-slurm` v2.59.1（`ae3d6877`）、sglang / Mooncake 上游仓库，以及 10-27 拉取的 InferenceX API（`evidence/inferencex-api-glm52-20261027.json`）。

修订记录：
- `c34cda69`：初版。
- `d614e55e`：cookbook 已收录 Dynamo；会话 header 只需设 AIPerf 环境变量；4+4 可以跨两台机器；同机两个 prefill 撞的是 ROCr 的 IPC socket；
  补了 e2e-norm 口径下与 ATOM PD 的对比。
- `842d132c`：记入 10-27 的三项决定（framework 名 `infera-sglang`、先不补 TTFT、所有 PD 形态一起提交）；按上游跟进 PR 重新评估 SGLang 补丁；
  `write_back` 保留（实测比 `write_through` 高 19.3%）并说明 NVIDIA 为什么不用补丁；补了镜像里的 Mooncake 版本、srt-slurm 的 frontend 与补丁规则、
  HiCache 撤栈显存释放在上游 SGLang 上的现状；第 4 节换成待做总表。
- 本版（10-28）：infera 怎么进容器（第 5.7 节）。infera 自己的 overlay（payload 挂载 + `infera-exec`）不适合 InferenceX；
  Dynamo 在 InferenceX 里是"原样镜像 + 启动时 pip 装固定版本的公开 wheel"；InferenceX 允许从 GitHub release 装 wheel（LMCache ROCm 先例）。
  D 组按此改写。
- 10-28（二）：记入与同事的分工、同事已完成的 PR 与复现指南、llr37 配置与 InferenceX 规则的冲突（第 5.8 节）；B 组标注负责人；
  新增 A6（上榜用哪套配置）、H8（KV 传输 1800 秒超时）；第 1 节加"本周开工的事"。

## 0. 结论

1. **InferenceX 不收外部数字，只发布它自己 CI 跑出来、经 PR 评审合入的结果。**
   "发布"的实际含义是：向 `SemiAnalysisAI/InferenceX` 提交一个 recipe PR，由 SemiAnalysis 的 `mi355x-amds` 集群（10 节点，Slurm + enroot，经 srt-slurm 拉起）
   用上游镜像重跑，sweep 全绿、eval 通过、AMD CODEOWNER 签 checklist、核心维护者批准后合入，合入时摄取这次 sweep 的结果。
   PR #210 的数字是在 crsuse2 上用 docker、自建镜像、手工脚本和 `918524ff` 客户端测的，**不能直接上榜，只能作为预期**。
2. **infera-sglang 按 SGLang 的规则审。** InferenceX 按底层推理引擎定规则（Dynamo 是部署层，`dynamo-sglang` 走 SGLang 规则），
   所以必须用上游 `lmsysorg` 镜像、不能打引擎补丁（要打就写 waiver）、配置要先进 SGLang cookbook。ATOM 能另走一套，是因为它本身是另一个引擎。
3. **最难的四件事**，按难度排序：
   - **引擎补丁。** 9 个补丁里 2 个可以去掉，3 个要先实测，4 个有 OPEN 的上游 PR 要推（第 5.1 节）。未合入的写进 waiver，InferenceX 只有 #2508 一个先例。
   - **HiCache 撤栈后显存释放慢。** 上游 SGLang 在 MI355X 上同样有这个问题，集群最多等 15 分钟，超时就让作业失败。
     我们每个 PP4 prefill 的主机内存池 1 TB，实测要 10–65 分钟，必须压到 15 分钟以内（第 5.5 节）。
   - **srt-slurm 要新增 infera frontend**，并且 SGLang cookbook 要收 infera router 的命令。两者都依赖外部评审。
   - **在 SemiAnalysis 集群上跑稳**，三种 PD 形态都要调通，而我们没有登录权限。
4. **镜像外依赖比预想的简单。** nightly 20261018 及之后的 `lmsysorg/sglang-rocm` 镜像自带的 Mooncake 与 ATOM PD 相同，功能齐全，不用重编；
   libionic 照同集群 Kimi-K3 vLLM recipe 挂载宿主机的 provider；会话 header 只需设 AIPerf 环境变量。
   infera 本身照 LMCache ROCm 的先例，从 Infera 的 GitHub release 按固定版本 pip 安装（要先发一个带 router 二进制的 release）；
   不用 infera 的 overlay，也不在集群节点上预放 payload（第 5.7 节）。
5. **分工（10-28）。** 同事负责 wheel 发版和他那部分的上游 PR（11 个 PR，5 个 Ready，都还没合入）；他的复现指南针对的是 llr37 配置，
   其中 KF kernel、FP8 草稿层、两个乱码开关在 InferenceX 上都不能用（第 5.8 节），上榜配置要重新定（A6）。
6. **仪表板口径。** 标准前沿用 E2E normalized interactivity P90（含 TTFT，下称 e2e-norm），每条曲线单独算。
   按 e2e-norm，我们 7 个点里 6 个在前沿；ATOM PD 的线在横轴约 51 以内都不低于我们（第 3 节）。按已定方针，先不为此补 TTFT。
7. **工期**：1–2 人，关键路径约 **8–10 周**；乐观 6–7 周，悲观 3 个月以上。最不可控的是上游 SGLang 补丁合入和集群上调通（只能经 CI 迭代，每轮以天计）。

## 1. 已定的决策（10-27）

| 决策 | 内容 | 影响 |
|---|---|---|
| framework 名 | 新增 `infera-sglang` | 要改 infx（`synthetic_acceptance.py` 的 `ENGINES`、`results/identity.py`、launch policy）并给 InferenceX-app 提 PR（`framework-aliases.ts` 等），多一轮跨仓库评审 |
| TTFT | 先不补 | 默认视图里 ATOM PD 的线在我们之上，接受这一点 |
| 首发范围 | 所有 PD 形态一起提交：1P1D（12 卡）、4+4（8 卡）、2P1D（16 卡） | 集群上要调通 2–3 台机器的三种布局；不再分期，也就不依赖 `append-only` |
| HiCache 写策略 | 保留 `write_back` | v0.5.22 上很可能不需要 staged write-back 补丁（10-28 更正，见 B5），要在 nightly 上确认 |

还没定的：上榜用哪套配置（A6，第 5.8 节）；srt-slurm 用静态 router 还是动态发现（第 5.3 节）；2P1D 的两个 prefill 同机还是各占半台（第 5.6 节）。

**分工与本周开工的事（10-28）**：同事负责 wheel 发版和他那部分的上游 PR（第 5.8 节）。我们这边不等决策就能开始的：

| 事项 | 产出 | 需要的资源 |
|---|---|---|
| InferenceX 集成骨架（F 组）：`infera-sglang` framework 注册、`infera-sglang.sh` 安装脚本与 `INSTALLERS` 登记、master 条目与 recipe 草稿 | 本地 InferenceX 分支，过 infx 自带的单元测试和 `infx generate` | 无需 GPU |
| 补丁必要性实测（B1 / B3 / B4 / B5 / B8）：原样 nightly 镜像 + 运行时装 infera，逐个去掉补丁 | 每个补丁"还需不需要"的结论，决定 waiver 范围 | 2 台 MI355X（138 / 140 正在跑 prefill 优化的队列） |
| HiCache 撤栈释放时间与大小的关系（H2） | 释放时间曲线，判断能否压进 15 分钟 | 1 台 MI355X |
| cookbook 草稿（G1） | MI355X 条目草稿 | 无需 GPU |

**10-28 进展**（三个分支都只在本地提交，未推送、未提 PR）：

| 仓库 / 分支 | 提交 | 内容 | 验证 |
|---|---|---|---|
| NVIDIA/srt-slurm `llying/infera-frontend`（基于 main `59920ea1`） | `9fde06e7`，**已提 Draft [srt-slurm #630](https://github.com/NVIDIA/srt-slurm/pull/630)**（10-28，fork 在 llying-001；写明依赖 Infera #222 / amd-infera 0.3.2，发版并在集群上跑通后转 Ready） | `frontend.type: infera`：`InferaFrontend`（继承 `DynamicFrontend`，worker 仍按 `direct` 处理）；`Frontend.direct_worker_launcher` 钩子让 SGLang worker 经 `python3 -m infera.engine.sglang` 启动并登记到 etcd；隐含 etcd、按角色数 `/v1/workers` 判断就绪、拒绝 srtctl 管理的 router 参数和外部 etcd；`configs/install-infera.sh`、`docs/infera.md`、example 与启动快照 | 新测试 10 项；全量测试失败名单与 main 逐条一致（11 个，均为容器环境）；ruff / 格式无新增；Python 3.10 下 `ty` 通过；`schema-docs --check`、`mkdocs --strict` 通过；example 在 mock orchestrator 下 exit 0 |
| AMD-AGI/Infera `llying/sglang-explicit-kv-events-config`（基于 main `319444f8`） | `47617d7a`、`bd17aaf1`、`7b6b65f1`，**已提 [Infera #222](https://github.com/AMD-AGI/Infera/pull/222)**（10-28；10-29 按 Copilot 评审补充：外部配置只接受 `tcp://*:<端口>`、topic 必须是 `kv-events`，三条行内意见已逐条回复；同时覆盖 vLLM 包装层） | 包装层沿用外部传入的 `--kv-events-config` 端口，不再自己找端口追加（srt-slurm 统一分配端口的规则要求）；外部没传时行为不变，仍按 DP rank 数自己分配 | `tests/unit/engine`、`tests/unit/kv` 504 项通过，ruff 通过 |
| SemiAnalysisAI/InferenceX `llying/infera-sglang`（10-29 rebase 到 main `98f0d6e3`，fork 在 llying-001） | `e6a34cfa`、`c9560348`，**已提 Draft [InferenceX#4629](https://github.com/SemiAnalysisAI/InferenceX/pull/4629)**（10-29；描述列出全部阻塞项，并按仓库规则说明补丁目录只收已合入的上游修复、#630 尚未合入） | `infera-sglang` 映射到 sglang；`infera-sglang.sh` 从 GitHub release 装 `amd-infera`（`--no-deps`，只补 `nats-py`）并登记进 `INSTALLERS`。骨架（以 #210 配置为底）：master 条目 `glm5.2-fp4-mi355x-infera-sglang-agentic-disagg`（nightly 20261018 按 digest 固定、`router: infera-router 0.3.2`、`dram-utilization: 0.85`）、recipe `glm5.2/infera-sglang/mi355x-fp4/agentx/disagg-variants.yaml`（1P1D 8 卡 decode C32/48/64/88、4+4 C64/80、2P1D C176）、perf-changelog（`pr-link` 用 XXX 占位）、`runners/srt-slurm/patches/630-infera-frontend.patch` | infx 全量测试失败名单与 main 一致、新增 3 项通过；补丁能叠加打在固定的 srt-slurm `ae3d6877` 上，infera 测试在该版本上通过；临时打补丁后 `infx generate` 生成并校验 7 个点，绑定结果正确（模型、镜像 digest、`ROUTER_VERSION`、模拟接受率 3.61、网卡引用） |

| sgl-project/sglang `llying/cookbook-glm52-mi355x-infera-pd`（fork 在 llying-001） | `a6f902f5`，**Draft [sglang#47117](https://github.com/sgl-project/sglang/pull/47117)**（10-28） | GLM-5.2 cookbook 新增 §3.6"MI355X 上用 Infera router 的 PD 分离"：镜像与安装、三种形态、etcd / 环境变量 / prefill / decode / router 的完整命令、关键设置的原因；Note 列出原样镜像仍需的 #21194、#46830、#33059、#46820 | Draft 条件：Infera 0.3.2 发版、原样镜像实测、InferenceX PR 提交后补链接 |
| SemiAnalysisAI/InferenceX-app `llying/infera-sglang-label`（fork 在 llying-001） | `4aa97624`，**Draft [InferenceX-app#1542](https://github.com/SemiAnalysisAI/InferenceX-app/pull/1542)**（10-28） | `FW_REGISTRY` 加 `'infera-sglang': { label: 'Infera SGLang' }` 和测试，照 TileRT #697；标题、提交、描述按该仓库要求双语并附 AI 披露 | constants 包测试通过；InferenceX PR 提交后转 Ready |

- **cookbook 的发现（10-28）**：GLM-5.2 页面的 "Agentic Long-Context Deployment" 一节是 NVIDIA 的同步脚本（sa-infx-cookbook，不在 SGLang 仓库里）从 InferenceX recipe 自动生成的，只覆盖 NVIDIA 硬件，渲染组件也只认 Dynamo / sglang-router，不能手改。
  AMD 的先例是在 "Advanced Usage" 里手写章节（DeepSeek-V4 页 §3.7 MI355X PD），所以我们写成 §3.6。SGLang 维护者是否接受文档化第三方 router 仍是风险（G2）。

| sgl-project/sglang `llying/cookbook-glm52-mi355x-infera-pd`（fork 在 llying-001） | `a6f902f5ea`，**Draft [sglang#47117](https://github.com/sgl-project/sglang/pull/47117)**（10-28） | GLM-5.2 cookbook §3.6：MI355X 上用 Infera router 的 PD 分离部署 | 只改文档；未跑 pre-commit |
| SemiAnalysisAI/InferenceX-app `llying/infera-sglang-label`（fork 在 llying-001，base `master`） | `4aa97624`，**Draft [InferenceX-app#1542](https://github.com/SemiAnalysisAI/InferenceX-app/pull/1542)**（10-28） | `FW_REGISTRY` 加 `'infera-sglang': { label: 'Infera SGLang' }` 与测试（照 TileRT #697）；标题、提交信息中英双语，附 AI 披露 | constants 包单测 75 项通过 |

| sgl-project/sglang `llying/cookbook-glm52-mi355x-infera-pd`（fork 在 llying-001） | `a6f902f5`，**已提 Draft [sglang#47117](https://github.com/sgl-project/sglang/pull/47117)**（10-28） | GLM-5.2 cookbook 新增 §3.6 "PD Disaggregation on MI355X with the Infera Router"：镜像与安装、三种形态、etcd / 环境变量 / prefill / decode / router 命令、关键设置说明，并注明原样镜像仍需 #21194、#46830、#33059、#46820。AgentX 一节是 NVIDIA 维护的同步脚本自动生成的（只覆盖 NVIDIA 硬件），所以照 DeepSeek-V4 页面 AMD 手写的 §3.7 写成手写小节 | Draft：等 Infera 0.3.2、原样镜像实测、InferenceX PR 提交后再转 Ready；PR 标题检查通过，测试闸门等维护者加 `run-ci`，Mintlify 预览等维护者批准 |
| SemiAnalysisAI/InferenceX-app `llying/infera-sglang-label`（fork 在 llying-001，base 是 `master`） | `4aa97624`，**已提 Draft [InferenceX-app#1542](https://github.com/SemiAnalysisAI/InferenceX-app/pull/1542)**（10-28） | `FW_REGISTRY` 登记 `'infera-sglang': { label: 'Infera SGLang' }` 并加测试（照 TileRT 的 #697）；标题和提交信息中英双语，附 AI 披露 | constants 包测试通过；Draft：等 InferenceX PR 提交后附链接；Vercel 预览等 SemiAnalysis 团队授权 |

依赖：srt-slurm 的 infera frontend 需要 Infera 带上 #222 的版本（文档写的是 amd-infera 0.3.2 或更新），要和同事的 v0.3.2 发版对齐。
另：138 上 docker 对 Docker Hub 的登录 token 已过期，拉不了 Docker Hub 上的新镜像（实测要用的 nightly 20261018 已在 138 本地，不受影响）。

- **与 Infera 版本的兼容性**：srt-slurm 把 `frontend.args` 原样传给 `infera-router`，以后新增的 router 策略和参数直接写在 recipe 里，srt-slurm 不用改；
  ≥ 0.3.2 的版本都能用，前提是这些约定保持兼容：可执行文件 `infera-router` 与模块 `infera.engine.sglang`、srtctl 管理的 router 参数、srtctl 传给包装层的参数、
  `/v1/workers` 的 `disagg_mode` / `status` 字段、#222 的端口行为。需要 worker 多开监听端口或改用 NATS 的新功能，才要同步改 srt-slurm。
- **srt-slurm #630 不一定要先合入（10-29 更正）**：作业启动时 `checkout.py` 先克隆固定版本的 srt-slurm、依次打上 `runners/srt-slurm/patches/*.patch`，
  再把打过补丁的版本装进 venv 跑 `srtctl apply`，所以带补丁就能跑。CI 工作流不调用 `infx generate`，矩阵规划（`variants.py`）也不导入 srtctl；
  只有本地查看用的 `infx generate` 从原样子模块导入 srtctl（此前把它写成"PR 校验"是错的）。剩下的是规则层面：补丁目录 README 写明"每个补丁回移一个
  上游修复"，未合入的 PR 属于灰色地带（ATOM #4045 的 604 补丁同样未合入），由 SemiAnalysis 维护者决定。
- **#630 的 router 改由 Infera 启动器起（10-28）**：Rust `infera-router` 只从本地路径读 tokenizer，recipe 用 HF 模型名时拿不到路径；改为
  `python3 -m infera.server --router-backend rust`（#210 kit 也是这样起的），由它解析 HF 名再 exec Rust router，默认值与 Rust 一致；
  `router-tokenizer-path` 默认取 worker 加载的模型。
- **recipe 骨架相对 #210 kit 的差异**（都要在原样 nightly 上实测确认）：KV 事件端口交给 srt-slurm 分配（不用 kit 的 `--kv-events-bind` / snapshot 端口）；
  Mooncake 用上游写法（`MC_DISABLE_HIP=1`、`MC_ENABLE_DEST_DEVICE_AFFINITY=1`，去掉对应我们补丁版的 `MC_ENABLE_DEST_LOCAL_RAIL`、`MC_IB_TIMEOUT`、
  `MC_DISABLE_HIP_TRANSPORT`）；网卡 `rdma0`–`rdma7`；不挂 libionic（同集群 DSv4 SGLang PD 用同类镜像跑 RDMA 也没挂）；未设 `ulimit -l`（H3）；
  `SGLANG_OPT_USE_TOPK_V2=true` 沿用 kit（包装层在 ROCm 上默认关它，原样镜像上要确认）；2P1D 两个 prefill 同机（A4 未定）。
- **为以后的 infera-vllm 预留（10-28）**：#630 的钩子改为 `direct_worker_launcher(backend, process, runtime)`，包装层模块按引擎查表 `WORKER_LAUNCHERS`
  （目前只有 sglang）；以后加 vLLM 只需加一行表项、让 vLLM backend 调用钩子，并处理 vLLM 每个 DP rank 各自登记的就绪计数。Infera #222 扩展到 vLLM 包装层
  （端口解析抽到 `infera/common/kv_events_config.py`，并拒绝 router 不订阅的 topic）。InferenceX 侧只需加 `infera-vllm` → `vllm` 映射，但规则按 vLLM 走
  （上游 vLLM 镜像、vLLM recipes 网站）。
- **InferenceX 的 PR 怎么拆**：InferenceX 仓库一个 PR（framework 注册 + 安装脚本 + recipe + master 条目 + perf-changelog，照 TileRT #3552 的做法），
  InferenceX-app 一个 PR（仪表板显示名）。framework 注册不单独提：单独提时评审看不到用法、CI 也覆盖不到，而且要和 srt-slurm 子模块升级（或带补丁）一起改。

## 2. InferenceX 的发布流程与先例

```text
我们的 PR（SemiAnalysisAI/InferenceX）
  ├─ configs/amd-master.yaml               新条目：镜像、模型、framework、每个点的拓扑与并发
  ├─ benchmarks/multi_node/srt-slurm-recipes/glm5.2/sglang/mi355x-fp4/agentx/disagg-variants.yaml
  │                                         srt-slurm recipe 片段：base + 每点一个 override
  ├─ perf-changelog.yaml                   末尾追加一条（触发 sweep 与结果收集）
  ├─ infx/ 代码（infera-sglang framework）、runners/srt-slurm/patches/（infera frontend）、docs/waiver/<PR>.md
  └─ 标签 full-sweep-fail-fast → mi355x-amds 上每点一个 Slurm 作业：起栈 → AgentX 3600 s → GSM8K eval
        → 全绿 → AMD CODEOWNER checklist → 核心维护者批准 → /use <run_id> → merge_with_reuse 合入并摄取结果
另需：SGLang cookbook PR（sgl-project/sglang）在 InferenceX PR 之前 MERGED；InferenceX-app PR（framework 名）
```

- **每点全新部署**；合成接受率由 driver 按 `infx/golden_al_distribution/` 注入（GLM-5.2 thinking_on K=5 → 3.61），recipe 里不能写死。
- **Eval**：每个拓扑在最高并发跑一次 agentic GSM8K（真实接受率），门槛默认 0.90。我们 PD example 实测 0.9666 / 0.9674。
- **AgentX 客户端**：用 InferenceX 当时 main 上的版本，不是我们固定的 `918524ff`。
- **会话 header**：InferenceX 默认不发 `X-Dynamo-Session-ID`（`runtime_settings.sh` 设为 `false`），由 recipe 的 benchmark env 选。
  vLLM 的 recipe 已有先例设 `AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID: '1'`，发的正是 infera router 认的 `X-Session-ID`。router 不用改。
- **规则原文**：lmsysorg 镜像见 [PR_REVIEW_CHECKLIST.md 第 61 行](https://github.com/SemiAnalysisAI/InferenceX/blob/7ca692b9afea00363297b5eab4ef9b5e72c982fe/inferencex-e2e/docs/PR_REVIEW_CHECKLIST.md?plain=1#L61)；
  不打补丁见同文件第 65 行；Dynamo 按底层引擎定规则见第 19–30 行；引擎分类见 `docs/MODELS.md` 第 66–86 行。

**先例怎么处理的：**

| | Dynamo+SGLang（GB300，[#4300](https://github.com/SemiAnalysisAI/InferenceX/pull/4300)，已合入） | Dynamo+TRT-LLM（GB300） | ATOM PD（MI355X，[#4045](https://github.com/SemiAnalysisAI/InferenceX/pull/4045)，OPEN） |
|---|---|---|---|
| 镜像 | lmsysorg nightly，digest 固定，零补丁 | NVIDIA 官方 `nvcr.io/nvidia/tensorrt-llm/release` | `rocm/atom-dev` nightly |
| 部署层 | Dynamo wheel 运行时安装（`dynamo.install`：从 PyPI 下载固定版本的 wheel，启动前 pip 装进容器） | 同左，固定版本 | AToMesh 在 ATOM 镜像里 |
| srt-slurm | 原生支持（srt-slurm 本来就是围绕 Dynamo 做的） | 原生 | `atomesh` frontend，由 SemiAnalysis 的 cquil11 提交（srt-slurm#433） |
| cookbook | sglang#45111 / #45311（已合入） | 不需要（非 vLLM/SGLang） | 不需要 |
| KV 传输 | — | — | 上游 Mooncake `78566d45`，只靠环境变量（`MC_DISABLE_HIP=1`、`MC_GID_INDEX=1`、`MC_MAX_MR_SIZE=1 GiB`） |
| DRAM KV offload | — | — | Mooncake Store，主机内存池在独立的 owner 进程里，用 2 MiB 大页 |
| 布局 | — | — | 每个 TP4 worker 独占半台（1P2D 用了 3 台） |

和我们最像的是 Dynamo+SGLang：上游镜像 + 运行时安装的部署层 + 写有部署层命令的 cookbook。差别是 Dynamo 有公开发布的 wheel、srt-slurm 原生支持，
infera 两样都还没有（infera 的 GitHub release 目前只有纯 Python 的 wheel，不含 router）。
同集群上 SGLang PD 的先例是 `dsv4-fp4-mi355x-sglang-disagg-agentic-umbp-dspark`（sglang-router frontend，MoRI 传 KV），集群写法可以照抄。

## 3. 按仪表板口径重算 PR #210 的 7 个点

仪表板的标准前沿用 e2e-norm（逐请求 E2EL / OSL 的 p90 取倒数，含 TTFT）。interactivity 等其他视图只显示"e2e-norm 前沿与该坐标下 Pareto 前沿的交集"
（`docs/MODELS.md` North-star policy）。前沿按每条曲线单独算（InferenceX-app `canonicalFrontier.ts` 按 `hwKey`、`precision`、`date` 分组，
`hwKey` 含硬件、framework、MTP），别家的点不会把我们挤掉。

`scripts/e2e_norm_table.py` 从 `../2p1d-v0522-20261020/results/` 读 AgentX JSON：

| 点 | GPU | tok/s/GPU | interactivity p90（PR 用的） | E2E normalized p90（仪表板） | TTFT p90 s | e2e-norm 下在前沿 |
|---|---:|---:|---:|---:|---:|---|
| 2P1D C176 | 16 | 38,392 | 43.2 | 14.4 | 9.5 | 是 |
| 1P + 4 卡 decode C80 | 8 | 37,310 | 44.3 | 14.9 | 10.8 | 是 |
| 1P + 4 卡 decode C64 | 8 | 32,532 | 48.6 | 24.6 | 4.2 | 是 |
| 1P1D 均衡 C88 | 12 | 31,088 | 71.7 | 14.1 | 11.5 | **否**（被 4+4 C80 支配，两种视图里都会消失） |
| 1P1D 均衡 C64 | 12 | 28,981 | 86.0 | 25.4 | 5.9 | 是 |
| 1P1D 均衡 C48 | 12 | 22,331 | 103.9 | 39.2 | 3.0 | 是 |
| 1P1D 均衡 C32 | 12 | 16,309 | 111.7 | 51.2 | 2.3 | 是 |

- e2e-norm 下的杠杆是 TTFT 尾部：prefill 饱和时的排队，以及 shortest-prefill-first 把长的冷首轮往后推。按已定方针先不优化。
- InferenceX API 不返回 e2e-norm 字段，现有 MI355X / B300 线在新口径下的位置要到仪表板上看；PR 里"比 MI355X 线高 1.4–2.2 倍"的对比要按 e2e-norm 重做。

**与 ATOM PD 对比**（#4045 在 mi355x-amds 上的 sweep 39028510364，`../gb300-atom-pd-analysis-20261021/evidence/atom-pd-pr4045/run39028510364/`）：

| ATOM PD 点 | tok/s/GPU | interactivity p90 | E2E normalized p90 |
|---|---:|---:|---:|
| 1P1D C96 | 42,884 | 42.1 | 14.6 |
| 1P1D C72 | 38,487 | 48.8 | 25.9 |
| 1P1D C56 | 31,067 | 58.4 | 32.5 |
| 1P2D C56 | 24,030 | 80.0 | 39.5 |
| 1P3D C24 | 10,073 | 146.8 | 60.2 |
| 1P3D C12 | 3,974 | 186.0 | 75.5 |

- 按 interactivity p90：横轴 50 以内 ATOM 更高；55–110 我们更高（1P1D C88 31.1k @ 71.7 对 ATOM 1P2D 24.0k @ 80.0）；再往右只有 ATOM 有点。
- 按 e2e-norm（默认视图）：ATOM 在横轴约 51 以内都不低于我们（C96 42.9k @ 14.6 对我们 38.4k @ 14.4），51 附近持平（约 16k），再往右只有 ATOM 有点。
- 我们的数字在 crsuse2 上测，ATOM 的在 SemiAnalysis 集群上测；#4045 自述两个集群差约 15%，我们到该集群上大概率还会更低。

## 4. 待做总表

级别：**P0** 发布的硬门槛；**P1** 必须做但不单独卡发布；**P2** 优化或参考。依赖里"内部"指我们自己。

| 编号 | 事项 | 现状 / 依据 | 下一步 | 依赖 | 工期 | 级别 |
|---|---|---|---|---|---|---|
| **A** | **对齐与决策** | | | | | |
| A1 | 提交方式与分工 | 未开始；fork PR 每次 push 都要维护者重新打标签 | 找 AMD 的 CODEOWNER（@seungrokj、@chunfangamd、@billishyahao 等）确认谁提 PR、谁签 checklist、谁打 sweep 标签 | AMD | 1 周 | P0 |
| A2 | infera 按"部署层"处理、新 framework 名、waiver 接受度 | infera-sglang 走 SGLang 规则；Dynamo+SGLang 是先例 | 问 SemiAnalysis 核心维护者 | SemiAnalysis | 与 A1 并行 | P0 |
| A3 | srt-slurm 接入方案 | **10-28 定：先走动态发现提 PR**，看评审推进情况；推进困难再改静态 router（第 5.3 节） | 实现 infera 动态 frontend 并提 srt-slurm PR | 你 | — | 已定 |
| A4 | 2P1D 两个 prefill 的放置 | 同机（2 台）或各占半台（3 台）（第 5.6 节） | 定方案，建议各占半台 | 你 | — | P1 |
| A5 | 集群访问 | 没有 mi355x-amds 登录权限 | 申请 1–2 节点交互使用，或拿到失败作业的日志 | AMD | — | P1 |
| A6 | 上榜用哪套配置 | PR #210 的配置合规问题少、数字低（4+4 C80 37,310）；同事的 llr37 / llr48 配置数字高，但含 InferenceX 禁止的项（第 5.8 节） | 定方案；建议以 #210 配置为底，逐项加入 llr37 里合规的参数（K3、decode mem 0.95 等）并实测 | 你 | — | P0 |
| — | 已定 | framework 名 `infera-sglang`；先不补 TTFT；三种 PD 形态一起提交；保留 `write_back` | — | — | — | — |
| **B** | **引擎补丁**（详见第 5.1 节） | | | | 开发 1–2 周；上游合入不可控 | |
| B1 | DP padded rows | sglang#33059 **Draft**（同事维护）；新 CI 上 Qwen3-8B PD 的 GSM8K 只有 1%，原因未查清 | 同事查 CI 失败、推评审；我们在 nightly 上实测是否仍必需 | 同事 + SGLang 维护者 | 不可控 | P0 |
| B2 | DSA DP 同步死锁 | #46830 **Ready for review**（同事，接替被自动关闭的 #33973） | 同事推合入 | 同事 + SGLang 维护者 | 不可控 | P0 |
| B3 | DSA page table 行数 | **10-28 更正：本配置不需要**。补丁说明记录 `--json-model-override-args '{"index_share_for_mtp_iteration":false}'` 可替代它，并已在 2×8 MI355X 上验证（4/4、32/32 两次、64/64）；#210 配置正好设了该参数 | 去掉补丁；只有以后想开 index_share 时才需要（第三方 #32209 修法不同） | — | — | P1 |
| B4 | 非末块 KV 等待事件 | 接收端 #41395 已合入；发送端同事不自己提，跟进他人的 #45524（Ready）/ #43299 | 我们在 nightly 上去掉补丁做长 prompt 正确性对比；需要就跟 #45524 | 我们 + #45524 作者 | 2–3 天 | P0 |
| B5 | HiCache staged write-back（ROCm 门控不一致） | **10-28 更正：v0.5.22 上很可能不需要**。补丁修的是 v0.5.16 的崩溃；v0.5.18 起控制器按池分别处理回写索引，MI355X 镜像上它只是让 MLA 保守地走非 JIT 路径（性能取舍，不是正确性依赖）；上游 main 上没有对应 bug，不适合提 PR | 去掉补丁，在 nightly 上确认 `write_back` 不崩；不带补丁时 MLA 走 JIT，prefill 满 chunk 慢的 13% 可能拿回 | 我们 | 实测时一并做 | P1 |
| B6 | KV 指标非阻塞发送 | #46820 **Ready for review**（同事） | 同事推合入 | 同事 + SGLang 维护者 | 不可控 | P0 |
| B7 | PP 中间级无 embedding | **10-29：已提替代 PR [sglang#47184](https://github.com/sgl-project/sglang/pull/47184)**（#21194 的改动 rebase 到当前 main，`Co-authored-by` 保留原作者 Xinyi Song；#21194 第一处 hunk 冲突、标题不符合新规则、3 月起无动静）。已在 #21194 下[告知](https://github.com/sgl-project/sglang/pull/21194#issuecomment-6486741914)，原作者若更新则关掉 #47184。同事（S13）不用再提 | 等维护者评审、加 `run-ci`；退路 prefill 改 TP4 | 我们 + SGLang 维护者 | 不可控 | P0 |
| B8 | MLA absorb BMM 配置 | 同事提了 issue #46829（MI355X 上仍能复现），不提绕过补丁的 PR | 我们在 nightly 上去掉补丁重测；仍崩就写 waiver | 我们 + 同事 | 1 天 | P0 |
| B9 | Responses tool 参数、decode radix | 本配置不需要 / 不起作用 | 去掉（确认 AgentX 走 chat completions） | 内部 | — | P1 |
| B10 | waiver 文件 | InferenceX 只有 #2508 一个先例：补丁脚本在启动前把修复打进容器源码，`docs/waiver/<PR>.md` 写所涵盖的配置、改了什么（上游 PR / issue 表）、原样镜像为什么跑不了、上游链接、移除计划；issue 链接可以，#2508 里还有一条无上游 PR 的本地改动 | 每个未合入的补丁一行（B1 #33059、B2 #46830、B6 #46820、B7 同事的 PR 或 #21194、B8 issue #46829，B4 视实测）；"为什么跑不了"要在原样 nightly 上复现取证；补丁脚本与 `infera-sglang.sh` 合用一个 `setup_script`（另写一个"安装 + 打补丁"脚本并登记进 `INSTALLERS`） | 内部 + CODEOWNER | 0.5 周 | P0 |
| B11 | override 参数审查 | 如果 B3 采用 `index_share_for_mtp_iteration:false` | PR 里说明只是关掉跨迭代复用，计算只多不少 | 内部 | — | P1 |
| **C** | **镜像外依赖：Mooncake 与网卡**（第 5.2 节） | | | | | |
| C1 | 镜像与 Mooncake | nightly 20261018 起自带 `78566d45`（与 ATOM PD 相同），功能齐全；正式版 v0.5.22 的 `01d1eb2a` 跑不了跨机 PD | 用 nightly 镜像、不重编、设 `MC_DISABLE_HIP=1` 实测 | 内部 | 2 天 | P0 |
| C2 | 两个 Mooncake RDMA diff | **10-29 核实：CI 集群不需要**。ATOM PD 在 `mi355x-amds` 上用上游 rail 匹配（`MC_ENABLE_DEST_DEVICE_AFFINITY`）跑通，1 小时无传输失败；AUS 不带它们也跑通；只有 crsuse2 因网卡丢帧需要（第 5.10 节） | 不带；推上游 #4540 / #4541 只为 crsuse2 这类网络 | 内部 + Mooncake 维护者 | — | P2 |
| C5 | HIP dma-buf | **10-29 核实：CI 集群走 peer 方式，不用 dma-buf**。ATOM PD 日志每个 worker 都是 "HIP dmabuf disabled"；AUS 上设 0 会段错误 | recipe 和 cookbook 补 `MOONCAKE_DISABLE_HIP_DMABUF=1`（#4629、#47117 待更新） | 内部 | 小 | P0 |
| C3 | libionic | 不进镜像；不用 `infera-exec` 覆盖镜像文件的做法（违反"不覆盖容器文件"） | recipe `extra_mount` 只读挂载宿主机 `libionic-rdmav34.so`（照 Kimi-K3 vLLM） | 内部 | 小 | P0 |
| C4 | ionic 驱动 / 固件版本 | 公开日志查不到 | 问集群负责人，或跑调试作业打印 `nicctl show version firmware`、`modinfo ionic` | AMD / SemiAnalysis | — | P2（只在判断网卡行为时需要；第 5.10 节已用 ATOM 日志确认 peer 方式和无丢帧） |
| **D** | **infera 打包、发版与安装**（第 5.7 节） | | | | 1 周左右 | |
| D0 | 交付方式 | 不用 overlay payload / `infera-exec`，不在集群节点预放 payload；从 GitHub release 按固定版本 pip 安装（LMCache ROCm 先例） | 按此执行 | — | — | 已定方向 |
| D1 | release 里带 router 二进制 | Infera 仓库公开，`release.yml` 每个 tag 把 wheel 传到 GitHub release；但 v0.3.1 只有纯 Python 的 `amd_infera-0.3.1-py3-none-any.whl`，没有 `infera-router`。**10-28 起由同事负责** | release 流程增加带 router 的平台 wheel（如 `manylinux_2_35_x86_64`，与 lmsysorg 镜像的 glibc 2.35 匹配）；PyPI 是可选项，不卡发布 | 同事 | 1 周 | P0 |
| D2 | 发版 | #210（10-26 合入）、#213（10-23 合入）的 router 改动都不在 v0.3.1 里（v0.3.1 落后 #210 合入点 30 个 commit） | 同事发 v0.3.2，tag 发布后不再改动；InferenceX 固定这个版本 | 同事 | 含在 D1 | P0 |
| D3 | 安装脚本 | — | 新增 `infera-sglang.sh` 并登记进 `infx/srt_slurm/workload.py` 的 `INSTALLERS`；`--no-deps` 安装，所需依赖逐个固定版本，不动镜像里的 torch / sglang / numpy | 内部 + SemiAnalysis | 含在 F3 | P0 |
| D4 | 包装层审计 | `kv_probe` 会 hook `RadixCache`（属 monkey-patching）、`kvd_wiring` 会改 `ServerArgs`，本配置都不启用；wheel 不带 SGLang 补丁 | PR 里说明包装层不改 SGLang 行为 | 内部 | — | P1 |
| D5 | 会话 header | 不用改 router | 设 `AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID: '1'` | — | — | 已解决 |
| **E** | **srt-slurm frontend**（第 5.3 节） | | | | 开发 1–1.5 周；评审 1–3 周 | |
| E1 | 上游 PR | ATOM 的 srt-slurm#433 由 SemiAnalysis 的 cquil11 提交，10 天合入 | 按 A3 的方案提交模块、测试、docs、example | 内部 + NVIDIA / SemiAnalysis | 见上 | P0 |
| E2 | InferenceX 里带补丁 | 规则允许回移上游修复，由 SemiAnalysis 核心维护者控制 | 先提上游 PR，再带补丁 | SemiAnalysis | — | P1 |
| **F** | **InferenceX 集成** | | | | 1.5–2.5 周 | |
| F1 | master 条目 | — | 镜像 digest、三种形态的拓扑与并发、`kv-offloading: dram` + hicache、router 版本；`dram-utilization` 与 HiCache 大小匹配 | 内部 | | P0 |
| F2 | recipe | — | 翻译 `worker.sh`：`@fabric.*` / `rdma0-7`、去掉 `SIM_ACCEPT_LEN`、会话 header、libionic 挂载、host_setup 钩子 | 内部 | | P0 |
| F3 | `infera-sglang` framework | 已定 | 改 infx `ENGINES`、`identity.py`、launch policy 和 `INSTALLERS`（D3）；master 写 `router: { name: infera-router, version: "0.3.2" }`；给 InferenceX-app 提 PR | 内部 + SemiAnalysis | 1–2 周 | P0 |
| F4 | 缓存统计 | ATOM 为此改过 infx（#3964） | 确认 PD 下只统计 prefill 端、HiCache 命中算对 | 内部 | | P1 |
| F5 | 其他 | — | perf-changelog、中英双语标题、AI 声明；`infx generate` + `srtctl dry-run` 验证 | 内部 | | P0 |
| **G** | **SGLang cookbook**（第 5.4 节） | | | | 1–2 周 + 评审 | |
| G1 | GLM-5.2 页面加 MI355X infera PD 部署说明 | **10-28 已提 Draft [sglang#47117](https://github.com/sgl-project/sglang/pull/47117)**：手写 §3.6（照 DSv4 页面 AMD 手写的 §3.7 MI355X PD），含镜像、infera 安装、三种形态、etcd / prefill / decode / router 完整命令、关键设置说明、仍需的 SGLang 修复（#21194、#46830、#33059、#46820）。AgentX 一节是 NVIDIA 的同步脚本（sa-infx-cookbook）从 InferenceX recipe 生成的，不能手改，且只覆盖 NVIDIA 硬件 | 等 Infera 0.3.2 发版、原样镜像实测、InferenceX PR 提交后转 Ready；必须在 InferenceX PR 前 MERGED | SGLang 维护者 | | P0 |
| G2 | 不收第三方 router 的退路 | — | cookbook 只写 SGLang 原生命令，router 链到 infera 文档（要 CODEOWNER 认可） | — | — | 风险 |
| **H** | **集群稳定性** | | | | 3–4 周 | |
| H1 | prefill GPU 内存访问错误 | 约 19 个实例小时 1 次（TP4 prefill），出现后要重置 GPU；PP4 下还没见到 | 提交前定位，或证明 PP4 长跑不再出现 | 内部 | | P0 |
| H2 | HiCache 撤栈后显存释放（第 5.5 节） | 上游 SGLang 同病；集群等 15 分钟，超时作业失败并 fail-fast；我们 1 TB / prefill 实测 10–65 分钟 | 测释放时间与 HiCache 大小的关系（先用 180 GB/rank）；定位根因提上游；退路缩池或挪到独立进程 | 内部 | | P0 |
| H3 | enroot 下锁定内存受限 | 容器 root 没有 `CAP_IPC_LOCK`（#4045 遇到过） | `ulimit -l unlimited`，照 #4045 验证 | 内部 | | P0 |
| H4 | 三种布局在集群上调通 | 1P1D、4+4 各 2 台；2P1D 2 或 3 台 | 首轮单点暴露问题，逐个形态调 | 内部 | | P0 |
| H5 | 同机两个 prefill（若 A4 选同机） | ROCr `@hsa<pid>` 撞名在 enroot 下大概率不存在；3P1D 时同机第二个 prefill 挂起，未查清 | 先在本地复现排除 | 内部 | | P1 |
| H6 | 本地模拟集群环境 | — | 自己节点上用 Slurm + enroot（非 docker、非特权、`ulimit -l` 受限）跑通 | 内部 | 1 周 | P1 |
| H7 | 权重与冷启动 | 集群没有预置模型；冷启动 10–15 分钟 + JIT | 节点本地 hub cache；调大 `health_check` 次数 | 内部 | | P1 |
| H8 | KV 传输 1800 秒超时 | 同事 llr44–48：decode 等 30 分钟收不到 prefill 的 KV，并发越高越多（C88 1 次、C104 12–17 次），关掉 PP 通信重叠后更多；prefill 侧原因未测 | 定位 prefill 为什么不发；CI 上一次超时就会拖长尾、可能让 eval 失败 | 我们 + 同事 | | P1 |
| **I** | **性能与口径** | | | | 1 周机时，可马上开始 | |
| I1 | 换 InferenceX main 的 AgentX 客户端重测 | 现在固定 `918524ff` | 重测三种形态的前沿点 | 内部 | | P1 |
| I2 | 原样上游镜像 + 运行时 infera 的性能 | 依赖 B、C | 测量与 PR #210 的差距 | 内部 | | P1 |
| **J** | **sweep 与评审** | | | | 2–3 周 | |
| J1 | sweep | 每点 2–3 台，集群 10 台，最多 3–5 点并行 | 首轮 `--trim-conc` 或单点，再 full-sweep-fail-fast | AMD 打标签 | | P0 |
| J2 | 签核与合入 | — | CODEOWNER 签 checklist → CI 校验 → 核心维护者批准 → `/use` → `merge_with_reuse` | AMD + SemiAnalysis | | P0 |

## 5. 关键事项说明

### 5.1 引擎补丁的上游状态（10-27 重新核对）

v0.5.22 镜像烘焙的补丁见 Infera `deploy/docker/patch.upstream.status.md`。10-27 dorado269 新提了一批 SGLang PR，其中两个是被关闭 PR 的跟进；
同日以"已被合入的 PR 取代"关闭了 #33970 和 #33968。

| 补丁 | 本配置需要吗 | 上游 PR 与状态 | 下一步 |
|---|---|---|---|
| `sglang_dsa/patch_dsa_indexer_hip_dp_padded_rows.py` | 需要（decode DPA） | [sglang#33059](https://github.com/sgl-project/sglang/pull/33059)（我们的），7-31 起 OPEN | 推评审 |
| `sglang_dsa/dsa_dp_sync.diff` | 需要（decode DPA） | [#33973](https://github.com/sgl-project/sglang/pull/33973) 是草稿，39 天未更新，10-03 被机器人关闭，无法重开 → [#46830](https://github.com/sgl-project/sglang/pull/46830)（10-27，rebase 到 main 并补回归测试）OPEN | 推 #46830 |
| `sglang_dsa/v0522/dsa_page_table_rows.diff` | 待实测 | [#32209](https://github.com/sgl-project/sglang/pull/32209)（第三方）OPEN，修法不同（裁剪 q / top-k） | 先测只用 `index_share_for_mtp_iteration:false`（gfx942 镜像先例）；不够再 waiver |
| `sglang_disagg/v0522/patch_prefill_nonfinal_chunk_wait_event.py` | 需要（正确性） | 接收端 [#41395](https://github.com/sgl-project/sglang/pull/41395) 10-06 合入；我们的 [#33970](https://github.com/sgl-project/sglang/pull/33970) 10-27 以"被 #41395 取代"关闭；但 10-19 对 10-18 基础镜像（已含 #41395）核对时，非末块 overlap 发送仍不记录完成事件 | 最新 nightly 上去掉补丁做长 prompt 正确性对比；仍需要就提新 PR |
| `sglang_rocm/patch_hicache_rocm_staged_write_back.py` | 需要（`write_back`） | [#28534](https://github.com/sgl-project/sglang/pull/28534) 7-09 合入但没覆盖 DSA pool；[#30350](https://github.com/sgl-project/sglang/pull/30350) 关闭；[#35043](https://github.com/sgl-project/sglang/pull/35043)（AMD-yanfeiwang）OPEN、有冲突，8-19 后无动静 | 见下文 |
| `sglang_kv_events/patch_kv_metrics_send_noblock.py` | 需要（kv-aware 路由） | [#46820](https://github.com/sgl-project/sglang/pull/46820)（10-27）OPEN | 推评审 |
| `sglang_pp/v0522/patch_deepseek_v2_pp_embed_dim.py` | 需要（PP4 prefill） | 第三方 [#21194](https://github.com/sgl-project/sglang/pull/21194) 修同一个 bug（`getattr` 兜底），3-23 起 OPEN，3 个 review 后无动静；[#28827](https://github.com/sgl-project/sglang/pull/28827) 关闭 | 在 #21194 上补 GLM-5.2 PP4 复现推合入；退路 prefill 改 TP4 |
| `sglang_rocm/v0522/patch_mla_absorb_bmm_default_config.py` | 需要实测：v0.5.21 起手写的 Triton tile 配置在 GLM-5.2 prefill 上触发 LLVM 断言崩溃，v0.5.22 未重测 | 无 | 最新 nightly 上去掉补丁重测；仍崩就提 issue / PR |
| `sglang_responses/v0522/patch_responses_unstreamed_tool_args.py` | 不需要（AgentX 走 chat completions，需确认） | 无 | 去掉 |
| `sglang_decode_radix_spec` | v0.5.22 上不起作用 | — | 去掉 |

v0.5.22 已不再烘焙、因上游已修的：HiCache 主机内存分配（[#35233](https://github.com/sgl-project/sglang/pull/35233) 合入，取代我们的 #33968）、
Mooncake 提前发送前的等待（#41395）。

**`write_back` 为什么在 NVIDIA 上不用补丁。** 问题只在 ROCm 上：

| | MLA KV pool（`pool_host/mla.py`） | DSA indexer pool（`pool_host/dsa.py`） |
|---|---|---|
| staged write-back JIT 门控（v0.5.22 与 main 相同） | `_is_cuda or _is_hip`（#28534 加了 HIP） | `_is_cuda` |
| CUDA | 开 | 开，两者一致 |
| ROCm | 开 | 关，两者不一致 |

- 在 v0.5.16 上，ROCm 的这种不一致会让控制器把 GPU 索引交给要求主机索引的 JIT kernel，prefill 第一次复用前缀就退出（−3）。
  v0.5.18 起控制器按 pool 分别搬索引，那种崩法不再必然出现，但门控不一致仍在；我们的补丁保守地让 MLA 也走非 JIT 路径。
- InferenceX 上 NVIDIA 的 GLM-5.2 SGLang 配置（GB300 Dynamo、B200 / B300 单机）用 `write_back`；所有 MI355X SGLang HiCache 配置都用
  `write_through` 或 `write_through_selective`，很可能就是绕开这个问题。
- 我们实测 `write_back` 比 `write_through` 高 19.3%（2P1D C176，25,695 → 30,659 tok/s/GPU，`../2p1d-aus-v0521-20261019/RESULT.zh-CN.md` §2.7），
  机制是缓存容量从"主机"变成"GPU + 主机"，prefill 新算量少 21%。同时 prefill 每个满 chunk 慢了 13%，因为 ROCm 上回写走的是非 JIT 路径。
- **路径**：先在最新 nightly 上不带补丁跑 `write_back`，看是否已经不崩；不行就推 #35043（它把 DSA indexer 等三个 pool 的门控都改成 `_is_cuda or _is_hip`，
  两边都走 JIT，可能还能拿回那 13%），或提一个让两边门控一致的最小修复。都不行就写进 waiver。
- **10-28 更正**：补丁自己的说明记录，v0.5.18 起控制器按池处理回写索引后，MI355X 镜像上它只是性能取舍、不是正确性依赖，所以 v0.5.22 上预期
  不带补丁也能跑 `write_back`，B5 降为"去掉补丁、实测确认"，不提 PR。

### 5.2 Mooncake 与网卡

从 Docker Hub 读镜像构建历史里的 `MOONCAKE_COMMIT`（不用拉镜像）：

| 镜像 | Mooncake commit | 说明 |
|---|---|---|
| `lmsysorg/sglang:v0.5.22-rocm720-mi35x`（正式版，digest `c4b61360…`，InferenceX GLM-5.2 单机线在用） | `01d1eb2a`（7-01） | wheel 版本 0.3.11.post1 |
| `lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261018`（我们的基础镜像）、`-20261019`（InferenceX DSv4 在用） | `78566d45`（9-07） | sglang#44404 在 10-18 升级："Bump Mooncake to v0.3.14-rc1 for cross-node KV transfer" |
| ATOM PD（#4045） | `78566d45` | 日志里显示 0.3.13 |
| 我们自建镜像重编的 | `faae8dd`（8-04）+ 两个 diff | 比 `78566d45` 落后 305 个 commit |

| 功能（我们构建脚本检查的） | 正式版 `01d1eb2a` | nightly `78566d45` |
|---|---|---|
| HIP dma-buf 接到 RDMA 传输 | 缺 | 有 |
| 大 MR 自动分片（Mooncake#2644）及分片 dma-buf 偏移 | 缺 | 有 |
| `MC_DISABLE_HIP`（跨机不走 HIP 传输） | 缺 | 有 |

- 用 nightly 20261018 及之后的镜像，Mooncake 不用重编；`Dockerfile.sglang` 里"基础镜像带的是更旧的 Mooncake"这句注释对这个基础镜像已不成立。
- **两个 RDMA diff 与驱动版本无关**，来自 cross-rank RCA（`../cross-rank-rca-20260921/REPORT.zh-CN.md`）：decode 端 ionic 网卡的 DMA 写被反压时会在网卡内部丢帧，
  PFC 拦不住。dest-local-rail 让写往 GPU j 的流量两端都走 `ionic_j`，ACK 超时设为 1.07 s。没有补丁时 Mooncake 会重试，不会崩。
  ATOM PD 在该集群上不带它们跑通，功能上不需要。
- **libionic**：Kimi-K3 vLLM 在该集群的 recipe 只读挂载宿主机 `/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so`
  （"Match the host's Ionic provider without replacing the image's verbs core"），照抄即可，不用知道驱动版本。
- **ionic 驱动与固件版本**：`check-rdma.sh` 只查设备与 QoS / DCQCN，公开日志不打印版本。只在判断"会不会丢帧"时需要。
- `MC_MAX_MR_SIZE=1 GiB` 沿用（ATOM 同）。如果改用 MoRI，Mooncake 相关问题都不存在，但要重新调传输参数并确认 infera 的 KV 功能不依赖 Mooncake。

### 5.3 srt-slurm 与 infera frontend

[NVIDIA/srt-slurm](https://github.com/NVIDIA/srt-slurm)（命令 `srtctl`）把一份 `schema: 2` YAML recipe 变成 Slurm 作业：分配节点、在容器里起 worker、
起 frontend 与附属服务（etcd、NATS、Mooncake master 等）、做健康检查、跑 benchmark。InferenceX 的多节点测试全部经它拉起。

- **frontend 机制**：`src/srtctl/frontends/` 下一个模块一种类型，`@register_frontend` 注册。两种基类：
  `StaticRouterFrontend`（router 从命令行拿 worker URL：sglang-router、vllm-router、smg、atomesh）和 `DynamicFrontend`（worker 自己注册，目前只有 Dynamo）。
  新类型要提交模块、`tests/test_<type>_frontend.py`、`docs/<type>.md` 和 `examples/` recipe。
- **Dynamo**：srt-slurm 本来就是围绕 Dynamo 做的，worker 有专门的 `dynamo` 启动方式（`python -m dynamo.sglang`），etcd / NATS 由 srt-slurm 起，
  Dynamo 由 recipe 的 `dynamo:` 块运行时安装。
- **非 NVIDIA 的 frontend**：ATOM / AToMesh（srt-slurm#433，+646 行，9-13 → 9-23）和 TileRT（#529，当天合入）都由 SemiAnalysis 的 cquil11 提交。
- **规则核对（10-28）**：srt-slurm 没有禁止其他动态 frontend。`DynamicFrontend` 的说明写明"Dynamo is the only implementation today. A subclass sets `type` and
  `worker_launch`…"，是可扩展的基类。唯一相关的约束是"A router binary that can also take static URLs … is a mode of a static router, not a dynamic frontend"：
  infera-router 目前只支持 etcd / Kubernetes 发现，按动态接入合规；若以后给它加静态模式，就要改走静态。动态方式的成本在工作量：worker 启动方式目前写死为
  `dynamo` / `direct`，按 "Names go in tables" 规则（`tests/test_design_rules.py` 自动检查）要改成表或钩子。
- **静态方式对功能和性能的影响（10-28 核对）**：router 与 worker 之间全走 SGLang 原生接口（HTTP 直连、各 DP rank 的 ZMQ KV 事件端口、`X-Data-Parallel-Rank`、
  bootstrap 端口、`/flush_cache`），router 的 `Worker` 字段都能从 `/get_server_info` 读到，所以本配置的功能与性能不受影响；唯一风险是包装层自动设的环境变量
  （`SGLANG_OPT_USE_TOPK_V2=0`、`MC_GID_INDEX`、目标设备亲和、dma-buf 路径、KV 网卡 IP、Mooncake 拓扑 JSON）要在 recipe 里写全。
- **10-28 决定：先按动态方式提 PR**，推进困难再改静态。两种方案的对比：

| | 方案 A：静态 router | 方案 B：动态发现 |
|---|---|---|
| 引擎怎么起 | srt-slurm 原样起 `sglang.launch_server` | 起 `python -m infera.engine.sglang`，由它拉起 SGLang 并注册到 etcd |
| srt-slurm 改动 | 小：照 `atomesh.py` 写一个子类 | 中：`DynamicFrontend` 的第二个实现，加上 SGLang 后端的新 worker 启动方式；违背"frontend 管就绪、后端管 worker 命令"的设计规则，评审阻力更大 |
| infera 改动 | 大：router 要支持静态 worker 列表；PD 启动探测、KV snapshot 等包装层功能用不上 | 小 |
| 与 PR #210 一致性 | 有差异，要重测 | 一致 |

- **在 InferenceX 里带补丁**：`runners/srt-slurm/patches/README.md` 规定这个目录由 InferenceX 核心维护者控制，每个补丁回移一个上游 srt-slurm 修复，
  上游合入后删除。这类补丁改的是测试框架，不需要 waiver。先例：`573-participating-gpus.patch`（上游已合入）；#4045 的 `604-atom-tp-from-pp.patch`
  （上游 OPEN，#4045 本身也未合入，"上游未合入就带补丁"能否被接受尚无定论）。做法是先提上游 PR，再带补丁。

### 5.4 SGLang cookbook

- checklist 明确允许 cookbook 写 Dynamo 这类部署层的安装、router / frontend 与引擎命令；用了部署层也不能豁免 cookbook。
- [GLM-5.2 页面的 Agentic Long-Context Deployment 一节](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.2#agentic-long-context-deployment)
  由 `docs/src/snippets/agentx/zai-org/glm-5.2.jsx` 渲染，sglang#45111（10-19）/ #45311（10-20）写入，按 InferenceX config 列出 `configKey`、recipe 路径、
  `dynamo.frontend` / `dynamo.sglang` 命令和 NGINX 配置。目前只有 GB300、GB200、B300、B200、H200，**没有 MI355X**，我们会是第一个 AMD 条目。
- 三种 PD 形态都要覆盖（"一个聚合示例不覆盖分离式部署"）；InferenceX 专用的设置（合成接受率、AIPerf）不用写，但要在 checklist 里说明理由。
- 风险是维护者是否愿意写 infera 这个第三方 router。退路是只写 SGLang 原生命令，router 链到 infera 文档，需要 CODEOWNER 认可。

### 5.5 HiCache 撤栈后显存释放

- **上游 SGLang 同样有这个问题。** 已合入的 GLM-5.2 FP4 MI355X SGLang 单机配置（TP4，HiCache 180 GB/rank，`write_through`），
  在 [#4506](https://github.com/SemiAnalysisAI/InferenceX/pull/4506) 的 sweep 40201277205 里：7 个 HiCache 作业全部出现
  "Schedulers still alive 15s after ShutdownReq" 和 "still alive after 10.0s SIGKILL"，srtctl 等 30 秒后放弃；3 个不开 HiCache 的作业都没有。
- **SemiAnalysis 的处理**：作业照常记为成功、节点交还；下一个作业的 host_setup 里有 AMD 的 `@seungrokj` 加的 `wait_for_amd_gpu_clean`
  （[#2490](https://github.com/SemiAnalysisAI/InferenceX/pull/2490)，8-04；[#2643](https://github.com/SemiAnalysisAI/InferenceX/pull/2643) 接入；现在在 `runners/srt-slurm/hooks/common.sh`），
  每 10 秒查一次显存，降到 10% 以下才继续，最多等 15 分钟，超时作业失败。超时已发生过多次，都按基础设施故障重跑：
  #3916（DSv4 SGLang）、#4078 / #4483（ATOM）、#4485（vLLM），残留 22–25% 显存，PR 里没有归因。
- **ATOM PD 没遇到**（sweep 39028510364 的作业都在 10 秒内清干净）：它的 DRAM 池由独立的 `mooncake_client` 进程持有
  （4 × 349 GiB，`MC_STORE_USE_HUGEPAGE=1`，2 MiB 大页由 host_setup 预留、teardown 收回），GPU 进程不锁定大块主机内存。
  同集群的 SGLang DSv4 PD recipe 也把 DRAM 层放进独立的 UMBP 服务进程（`UMBP_DRAM_USE_HUGEPAGES=1`）。
- **对我们**：GLM-5.2 单机是 720 GB 池，释放基本能在 15 分钟内完成；我们每个 PP4 prefill 1 TB（2P1D 同机时一台 2 TB），在 crsuse2 上实测 10–65 分钟。
  超过 15 分钟，同节点下一个作业（包括别人的 AMD 作业）会失败，还会经 fail-fast 连带取消同 sweep 的其他点。
- **做法**：测释放时间与 HiCache 大小的关系（先用 180 GB/rank 对照）；定位根因（怀疑与锁定主机内存的解锁有关，未证实），修好提上游，所有 AMD SGLang HiCache 配置受益；
  退路是缩小主机内存池，或把 DRAM 层挪到独立进程（HiCache 的 mooncake / UMBP 存储后端）。

### 5.6 PD 形态与放置

| 形态 | GPU | 布局 | 台数 |
|---|---:|---|---:|
| 4+4 | 8 | PP4 prefill 与 4 卡 decode 各占一台的 4 卡，与实测一致 | 2 |
| 1P1D | 12 | PP4 prefill 占一台的 4 卡，8 卡 decode 占一台 | 2 |
| 2P1D | 16 | 两个 prefill 同机（与实测一致）或各占半台；8 卡 decode 一台 | 2 或 3 |

- ATOM PD 在该集群上每个 TP4 worker 独占半台，分母只算用到的 GPU，半台布局有先例。
- 2P1D 同机时：ROCr 在网络命名空间里绑定抽象 socket `@hsa<pid>`。docker 用 host 网络，两个容器共用这个命名空间，但各自 PID 从 1 开始，scheduler 会撞名；
  我们让 prefill-1 在 exec 前 fork 3 万个空 subshell，把 PID 推到 30000 以上。enroot 默认不建 PID 命名空间，PID 全局唯一，大概率不再撞名，烧 PID 的写法可以去掉。
  另外 3P1D 时同机第二个 prefill（PID 偏移 60000）挂起过三次，原因未查清（`../3p1d-tp4-tp8-v0521-20261012/RESULT.zh-CN.md`）。
- 建议 2P1D 两个 prefill 各占半台：没有撞名与挂起的风险，单台主机内存池也只有 1 TB（第 5.5 节）。代价是每点多占一台机器。

### 5.7 infera 怎么进容器：打包、发版与安装（10-28 调研）

**结论**：不用 infera 自己的 overlay，也不在集群节点上预放 payload；照 LMCache ROCm 的先例，从 Infera 的 GitHub release 按 master 配置的固定版本 pip 安装。

**infera 的 overlay 是什么**（`deploy/overlay/`）。payload 镜像 `infera-overlay` 装 infera / kvd（纯 Python）、`infera-router`（Rust）、
重编的 Mooncake 及其约 40 个依赖库和 `infera-exec`；把 payload 挂进原样的上游镜像，入口换成 `/payload/bin/infera-exec`，按容器的 Python 版本与
ROCm 主版本选目录、设 `PYTHONPATH` / `LD_LIBRARY_PATH` 后 exec 原命令。它在 k8s 上靠 initContainer 把 payload 拷进 Pod。除此之外，`infera-exec` 还会：

| `infera-exec` 的动作 | InferenceX 规则（`PR_REVIEW_CHECKLIST.md` 第 65 行） |
|---|---|
| 启动时给 SGLang 打补丁（`sglang_disagg`、`sglang_rocm`、`sglang_dsa` 等） | 禁止（"inline patches…in-place edits of site-packages, monkey-patching"），除非有 waiver |
| 用 `cp` 把宿主机 libionic 覆盖到镜像的 `libionic.so.1` | 禁止（"overwriting container files"） |
| 用我们重编的 Mooncake 替换镜像自带的 | 禁止（"installing forked/rebuilt engine wheels on top of the pinned image"） |

另外 payload 里的补丁集只对应一个 SGLang 版本、不做版本检查（`Dockerfile.payload` 里自己标了 TODO）；已发布的 `inferaimage/infera-overlay:v0.2.2` 只带 vLLM 补丁。

**Dynamo 在 InferenceX 里怎么进容器**（srt-slurm `ae3d6877`）。不是挂载 payload，而是：recipe 写 `dynamo.install: true` 和固定版本
（如 `source.wheel: 1.6.0.dev20261016`）→ srtctl 从 `pypi.org` / `pypi.nvidia.com` 下载这个版本的 `ai-dynamo`、`ai-dynamo-runtime`
（`runtime_scripts/dynamo_wheels.py`），挂进容器 `/srtctl-wheels` → worker 启动前 pip 安装 → `python -m dynamo.sglang`（同一进程里构造
`sgl.Engine`）或 `python -m dynamo.trtllm`（同一进程里调 `tensorrt_llm.LLM`）。NVIDIA 集群上的 `dynamo-wheels` 卷只是下载缓存。
思路和 overlay 一样（镜像不动、部署层后加），交付方式不同。

**InferenceX 允许的做法**：
- 镜像原样，启动前用 setup 脚本装**公开发布、固定版本、在 master 里登记**的组件。`infx/srt_slurm/workload.py` 的 `INSTALLERS` 登记这些脚本，
  框架把 master 里 `router` 或 `kv-offload-backend` 的 `version` 作为 `ROUTER_VERSION` / `KV_OFFLOAD_BACKEND_VERSION` 传进去，
  没写版本就失败（`RECIPES.md` 第 59 行）。`version` 必须是确切的发行版、包、wheel 版本或源码 commit，不能填镜像名（`configs/CONFIGS.md`）。
- router 用单独的公开镜像（srt-slurm `frontend.container_image`，按 digest 固定）。

**各 backend 的先例**：

| 组件 | 集群 | 怎么进容器 |
|---|---|---|
| Dynamo（SGLang / TRT-LLM / vLLM） | NVIDIA | `dynamo.install`，从 PyPI 装固定版本的 wheel |
| vllm-router | MI355X 等 | `vllm-router.sh`：`pip install vllm-router==$ROUTER_VERSION`；Kimi-K3 MI355X 用 `vllm/vllm-router` 镜像（digest 固定） |
| TileRT（GLM-5.3） | MI355X | `glm5.3-tilert-rocm.sh`：`pip install tilert==$ROUTER_VERSION`，缺 Mooncake 则 pip 装公开版；router 用 `ghcr.io/tile-ai/tilert-rocm-decode:0.1.6` |
| **LMCache（ROCm）** | **MI300X** | **`lmcache-mp-rocm.sh`：从 GitHub release 装 wheel** |
| Mooncake | NVIDIA（vLLM、Kimi-K3 B300） | `vllm-mooncake.sh`：`--force-reinstall` 换成 PyPI 上的 `mooncake-transfer-engine-cuda13==$KV_OFFLOAD_BACKEND_VERSION` |
| ATOM / AToMesh | MI355X | 都在 `rocm/atom-dev` 镜像里，frontend 也用这个镜像 |

替换镜像自带的组件也可以（Mooncake 那一行），前提是换成公开发布的版本，不能是自己重编的。

**集群节点上预放 payload**：规则没有逐字禁止，但没有一个 backend 这么做软件。节点上预先放好的只有模型权重（`models.entries`、`/it-share/data`，
包括 Kimi-K3 的 DSpark 草稿模型权重）、HF / aiperf 缓存、从公开 registry 导入的 squash 镜像、宿主机系统自带的 libionic（只读挂载）。
手工放的 payload 没有公开来源和版本可查，CODEOWNER 无法核对；我们也没有集群写权限。

**GitHub release 的先例**：已合入的 `minimaxm3-fp8-mi300x-vllm-agentic-mtp`（`amd-master.yaml` 第 1096 行）写
`kv-offload-backend: { name: lmcache, version: "0.5.3" }`，`lmcache-mp-rocm.sh` 用
`--find-links "https://github.com/LMCache/LMCache/releases/expanded_assets/v${KV_OFFLOAD_BACKEND_VERSION}-rocm"` 安装，
对应资产 `lmcache-0.5.3-cp312-cp312-manylinux_2_35_x86_64.whl`（10-28 可访问）。LMCache 的 ROCm 版不在 PyPI 上，可见 InferenceX 不限定 PyPI。

**infera 的现状**：`AMD-AGI/Infera` 是公开仓库，`release.yml` 每个 tag 都把 wheel 和 sdist 上传到 GitHub release（PyPI 那一步因为没配 token 一直跳过）。
最新的 `v0.3.1`（10-26）只有 `amd_infera-0.3.1-py3-none-any.whl` 和 sdist，不含 `infera-router`；PR #210 的 router 改动也还没发版。

**要做的**（待做总表 D 组）：
1. release 流程增加带 `infera-router` 的平台 wheel（如 `manylinux_2_35_x86_64`，与 lmsysorg 镜像的 Ubuntu 22.04 / glibc 2.35 匹配）。也可以把 router 二进制
   单独作为 release 资产，但 wheel 更规范。
2. 发 v0.3.2，包含 #210 的 router 改动，tag 发布后不再改动。
3. InferenceX 侧新增 `infera-sglang.sh` 并登记进 `INSTALLERS`，照 `lmcache-mp-rocm.sh` 写：
   `pip install --no-deps "amd-infera==${ROUTER_VERSION}" --find-links https://github.com/AMD-AGI/Infera/releases/expanded_assets/v${ROUTER_VERSION}`，
   所需依赖逐个固定版本列出，不动镜像里的 torch / sglang / numpy；master 写 `router: { name: infera-router, version: "0.3.2" }`。和 framework 名的改动（F3）放在同一个 PR。
4. wheel 里不带 SGLang 补丁；引擎侧 `infera.engine.sglang` 是纯 Python，起 `sglang.launch_server` 子进程，不改引擎。
   `kv_probe.py` 会 hook `RadixCache`、`kvd_wiring.py` 会改 `ServerArgs`，只在 probe 传输和 kvd 后端时启用，本配置用 zmq KV 事件，不触发；PR 里要说清楚。
5. 其余镜像外依赖：Mooncake 用 nightly 镜像自带的（第 5.2 节）；libionic 照 Kimi-K3 只读挂载宿主机文件。

PyPI 是可选项：GitHub release 现在就能用，不用等 PyPI 账号和权限。

### 5.8 同事（yihou）的进展与 llr37 配置的合规性（10-28）

**分工（10-28）**：同事负责 Infera 的 wheel 发版（D1 / D2）和他优化部分的上游 PR；其余（补丁实测、srt-slurm、InferenceX 集成、cookbook、集群调通）由我们做。

**同事已完成的**（`yihou/glm52.patch-inventory_20261027/`：`PR_SUMMARY_zh-CN.md`、`PR_PACKUP_MAP.md`、`PUBLIC_REPRODUCE.md`，10-28 05:17 提交 `e0dd1453`）：
1. **上游 PR**：自有核心 PR 11 个，5 个 Ready for review、6 个 Draft，**都还没合入**；另提 2 个 issue。与本配置相关的：

   | 本配置用到的补丁 | 同事的进展 |
   |---|---|
   | B1 DP padded rows | sglang#33059 Draft；新 CI 上 Qwen3-8B PD 的 GSM8K 只有 1%，原因未查清 |
   | B2 DSA DP 同步 | #46830 Ready |
   | B6 KV 指标非阻塞 | #46820 Ready |
   | B8 MLA absorb 崩溃 | issue #46829，不提绕过补丁的 PR |
   | B7 PP 非首级 hidden size（S13） | 没提，等你同意署名；他的文档没提到第三方 #21194 |
   | B4 非末块等待事件 | 跟进他人的 #45524（Ready）/ #43299 |
   | B3 page table 行数、B5 staged write-back | 不在他的范围内 |

   与本配置无关的：#46822（空闲 ROCm rank 的 indexer）、#45484（OffloaderV2）、Infera #218（只改镜像构建检查，Ready）、DCP 系列 4 个 Draft（#46819 / #46824 / #46826 / #46828）、
   Mooncake #5005（Draft；我们用 nightly 自带的 Mooncake 后用不到）、issue #46831（decode 保留显存上涨）。另在 #42163、#45156、#45337、#44322、Mooncake #4340 上补充了证据。
2. **公开复现指南**：针对他的 4+4 配置 llr37（C104，48,713 tok/s/GPU，模拟接受率 2.99），固定了基础镜像 digest、Infera `fdbafb60`、模型与 trace 的 revision，
   提供组装脚本 `repro/prepare_llr37.py` 和 `SHA256SUMS`，区分烘焙进镜像的补丁和启动时挂载的文件。只做了 CPU / 只读验证，没有重跑 GPU。
3. **补丁 ↔ PR ↔ 实验对照表**。

**llr37 配置与 InferenceX 规则的冲突**（复现指南用的是自建镜像 + 启动时挂载，这条路在 InferenceX 上走不通）：

| llr37 里的项 | InferenceX 规则 | 结论 |
|---|---|---|
| K01 KF kernel（`sitecustomize` 在运行时替换 sglang / aiter 的函数） | 第 65 行禁止 monkey-patching | 不能用；也没有上游 PR |
| A01 两个 aiter GEMM 调优 CSV（挂进 aiter 安装目录） | 改镜像文件 | 先合进 aiter 上游，或写 waiver |
| S13 替换整个 `deepseek_v2.py` | 引擎补丁 | 同 B7 |
| FP8 草稿层（`SGLANG_GLM_NEXTN_MOE_PTPC=1`） | 第 56 行：草稿层按发布精度跑，不能在提交侧量化降精度。`amd/GLM-5.2-MXFP4` 的第 78 层 NextN 不在 quark 量化范围里，按 BF16 发布 | 不能用（前提是这个开关默认关闭，未核实默认值） |
| prefill `SGLANG_PP_COMM_OVERLAP=1`、decode custom AR | 真实接受率下乱码，eval（GSM8K ≥ 0.90）过不了 | 必须关；关掉后最好是 llr48 C88 44,245（单次，正确性未确认） |
| `index_share_for_mtp_iteration=true` | 第 59 行：不接受减少模型 FLOPs 的优化 | 需在 PR 里说明来源，评审可能会问；#210 的配置显式设为 `false`，不受影响 |
| MTP K=3 | GLM-5.2 thinking_on 的 golden 接受率有 K=3（2.99） | 可以用 |
| decode mem 0.95、`--stream-interval 30`、`SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=8`、`--pp-async-batch-depth 1` | 普通参数 | 可以用，要写进 cookbook |
| prefill HiCache 250 GB/rank × 4 | DRAM 预算：`dram-utilization: 1.0` 时每个 4 卡服务器 1,499 GB；HiCache 另有约 23% 的 DSA indexer 池，250 GB/rank 实占约 1.23 TB | 在预算内；上限约 300 GB/rank |

所以要上榜，配置要在 #210 和 llr37 之间重新定（A6）：以 #210 的配置为底，加入 llr37 里合规的参数，在原样 nightly 镜像上用真实接受率验证正确性后重测。

### 5.9 InferenceX 上的 CI 怎么触发（10-29）

合入 Infera #222、发 v0.3.2 是必要条件，但不能单独触发、也不能单独跑通 #4629 的 CI：

1. **触发要人工打标签，fork PR 门槛更高。** sweep 只在 PR 被打上主标签（通常 `full-sweep-fail-fast`）时启动，打标签需要 InferenceX 写权限。
   #4629 来自 fork（`llying-001/InferenceX`），fork PR 拿不到仓库密钥，走"受信任的外部 fork 派发"（`trusted-external-sweep.yml`）：有写权限的人
   打标签只批准当前那个提交，"The PR must be open and non-draft"，之后每次推送都要重新打标签。同仓库的 PR 在 Draft 状态下也能打标签跑 sweep，
   所以最好请 AMD 有写权限的同事把分支推到 `SemiAnalysisAI/InferenceX` 主仓库（对应 A1）。
2. **现在触发也会失败。** 20261018 原样镜像缺 B7（#47184，PP4 prefill 非首级起不来）、B1 / B2（gfx950 上 DP attention 的 decode）、B6；
   B8 是否复现未知。要么这些修复合进上游并换更新的镜像，要么在 PR 里加补丁脚本和 waiver。recipe 里还有未验证项（第 5.10 节、HiCache 撤栈释放）。
3. **v0.3.2 要带 `infera-router` 二进制**，并包含 #213、#222（同事发版）。
4. srt-slurm #630 不用先合入：作业启动时会打上 `runners/srt-slurm/patches/` 里的补丁（第 4 节进展表下方的更正）。

建议顺序：先在我们自己的机器上用原样镜像 + 补丁脚本实测（SemiAnalysis 集群上每轮要重新打标签、以天计）→ 把实测确认还离不开的补丁做成补丁脚本和 waiver 加进
#4629 → #222 合入、发 v0.3.2（可并行）→ 请 AMD 有写权限的同事打标签，或把分支推到主仓库。

### 5.10 Mooncake 在 CI 集群上要不要 dma-buf 和那两个 RDMA 补丁（10-29）

**结论：都不需要。** 但 recipe 要补 `MOONCAKE_DISABLE_HIP_DMABUF=1`（目前没写，建议加进 #4629 和 cookbook #47117）。

| 证据 | 内容 |
|---|---|
| ATOM PD 在 SemiAnalysis `mi355x-amds` 上的日志（sweep 39028510364，节点 `mia1-p01-g09/g14/g15`） | 每个 worker 的 Mooncake（上游 `78566d45`）都打印 "HIP dmabuf disabled via MOONCAKE_DISABLE_HIP_DMABUF"，GPU 显存照样注册成功、KV 正常传输——即主机支持不走 dma-buf 的 GPU 显存注册（peer 方式）。这个开关不在 ATOM 的 recipe / sbatch 里，是 ATOM 镜像或代码默认设的 |
| 同一日志的 rail 选择 | "Auto-discovered Mooncake matched rails: primary=rdma1 rails=[rdma0..rdma7]"，每个 GPU 选同名网卡、GID index 1；配合上游的 `MC_ENABLE_DEST_DEVICE_AFFINITY=1`，在 rail 隔离的网络上两端走同名网卡。这是上游已有的功能，覆盖了我们 dest-local-rail 补丁要解决的跨 rank 选卡问题 |
| 同一日志的传输错误 | 1 小时里只有启动时 4 次 "Failed to retrieve segment descriptor"，没有传输失败、重试超时或 work completion 错误。crsuse2 上为之打 ACK 超时补丁的丢帧问题，在该集群上没有出现 |
| 我们在 AUS（`smci355-ccs-aus-*`）的实验 | `MOONCAKE_DISABLE_HIP_DMABUF=1` 是必需的：设 0 时 prefill 起栈即 scheduler 段错误（`2p1d-aus-v0521-20261019/RESULT.zh-CN.md` §2.1），设 1 后正常（日志 "HIP dmabuf disabled"）；`glm52.1p1d.c80.aus.packup_20260922` 用的是 HIP transport 与 HIP dma-buf 都关、每 rank 过滤到同号 ionic，没有那两个补丁也跑通 |
| crsuse2 | dma-buf 可用（=0），且 decode 端 ionic 在反压下丢帧，需要两个补丁（`cross-rank-rca-20260921`）——与 SemiAnalysis 集群、AUS 都不同 |

- 所以 dma-buf 和两个补丁都跟网卡驱动 / 固件和网络拓扑有关，是 crsuse2 特有的；SemiAnalysis 集群和 AUS 一样走 peer 方式、rail 隔离、无丢帧。
- `MC_MAX_MR_SIZE=1 GiB` 原本是为 ionic 拒绝 2 GiB dma-buf 注册而设，走 peer 方式后可能不再必要；ATOM 的 decode 也设了 1 GiB，保留无害。
- 仍查不到的：SemiAnalysis 节点的 ionic 驱动 / 固件版本（`check-rdma.sh` 只打印"QoS/DCQCN preflight passed"）。

### 5.11 实测方案（10-29 建议）

| | AUS（`smci355-ccs-aus-*`） | crsuse2 的 Slurm 分区（`amd-spur` / `amd-aifw-sglang`） | crsuse2 的 138 / 140（docker） |
|---|---|---|---|
| 与 SemiAnalysis 集群相像程度 | **最像**：ionic、rail 隔离、必须关 dma-buf（peer 方式），不需要那两个补丁 | 网卡行为同 138/140（dma-buf、丢帧），与 SemiAnalysis 不同 | 同左 |
| 调度与容器 | Slurm（`salloc` / `sbatch`）+ 节点上 docker；pyxis 未确认 | Slurm + pyxis（`srun --container-image` 可用），最接近 srt-slurm 的运行方式 | 直接 docker，无 Slurm |
| 适合测什么 | 原样镜像能不能跑、哪些 SGLang 补丁还离不开、正确性、三种形态的性能、HiCache 撤栈释放时间 | srtctl + infera frontend 的端到端（#630 实机验证），不追求性能代表性 | 快速排查，但网卡行为与 CI 集群不同 |

**建议两步走**：
1. **AUS 上做主要实测**：20261018 原样镜像，运行时装 infera（v0.3.2 之前先用分支自己打带 router 的 wheel），只加补丁脚本（逐个去掉 B1 / B2 / B4 / B6 / B7 / B8，
   确认 B3、B5 可去掉），按 recipe 参数 + `MOONCAKE_DISABLE_HIP_DMABUF=1`、不用两个 Mooncake 补丁；验证 GSM8K ≥ 0.90（真实接受率）、三种形态数字、
   撤栈释放时间。需要 AUS 上 2 台（2P1D 同机放置时）或 3 台。
2. **crsuse2 的 Slurm 分区上跑 srtctl 端到端**：用 #630 的 infera frontend 起一个小的 1P1D（可先用 Qwen 小模型，再上 GLM-5.2），验证 srt-slurm 侧的
   启动、端口分配、etcd 登记、就绪判断和 router 命令；这一步不看性能，主要给 #630 转 Ready 提供实机证据。

## 6. 时间线（1–2 人，按周）

```text
周          1    2    3    4    5    6    7    8    9   10
A 对齐      ██████████(持续沟通)
I 重测      ██████████
B 补丁实测  ██████████ ─ ─ ─ ─ ─ ─ 上游评审（不可控，waiver 兜底）─ ─ ─ ─ ─ ─
C Mooncake  █████
D infera         ██████
E srt-slurm      ██████████ ─ ─ 上游评审，期间带补丁 ─ ─
H2 显存释放 ███████████████
F 集成                ███████████████
G cookbook            ██████████████████
H 集群调通                      ████████████████████
J sweep/合入                                        ███████████████
里程碑：第 3 周 本地原样镜像 + 运行时 infera 跑通   第 6 周 PR + 首次 sweep   第 9–10 周 合入
```

- 关键路径：A1/A2 → B / C / D / E 开发（约 2 周）→ F + G（约 2 周）→ H 经 CI 调通三种形态（3 周）→ J（2 周），合计约 8–10 周。
  比初版多约 1 周，因为三种形态一起提交、要调通 2–3 台的布局，并新增 `infera-sglang` framework 的跨仓库改动。
- 乐观 6–7 周：上游补丁合入或 waiver 被接受，H2 在小幅缩池后就满足 15 分钟，集群上一两轮跑通。
- 悲观 3 个月以上：维护者要求先合上游、H2 压不下来、内存访问错误在 CI 上出现、cookbook 不收第三方 router。

## 7. 风险汇总

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| 上游 SGLang 补丁迟迟不合入（#33059、#46830、#46820、#21194） | 高 | 只能靠 waiver，可能被拒 | 用实测压缩必需补丁集；每个补丁先有上游 PR；waiver 写清移除计划 |
| HiCache 撤栈显存释放超过 15 分钟 | 高 | 高（弄坏同节点后续作业、fail-fast 连带取消） | 定位根因提上游；缩池；挪到独立进程 |
| prefill 内存访问错误在 CI 上出现 | 中 | 很高（要重置 GPU，影响所有 AMD 作业） | 提交前定位，或 PP4 长时间无复现的证据 |
| 集群上数字比 crsuse2 低（集群差约 15%，加上客户端、去补丁） | 高 | 中 | I1 / I2 先在本地按 CI 口径重测，对外只引用 CI 数字 |
| 默认视图里 ATOM PD 的线在我们之上 | 已知 | 中 | 按已定方针先不补 TTFT；突出"SGLang 是 GLM-5.2 的主引擎" |
| SGLang cookbook 不收 infera router 命令 | 中 | 高（checklist 过不去） | 照 sglang#45111 / #45311 的格式写，早沟通；退路链 infera 文档 |
| 维护者不接受新 framework 名 `infera-sglang` | 中 | 中 | A2 先确认；对照 `dynamo-sglang` 说明 |
| srt-slurm 不接受 infera frontend | 低–中 | 中 | 在 InferenceX 里带补丁（要核心维护者批）；方案 B 被拒就退到方案 A |
| 没有集群登录权限，调试慢 | 高 | 中（拉长工期） | 本地模拟 Slurm + enroot；请 AMD 团队协助拿日志或节点 |

## 8. 依据与来源

| 内容 | 位置 |
|---|---|
| 提交与评审规则 | InferenceX `CONTRIBUTING.md`、`inferencex-e2e/docs/PR_REVIEW_CHECKLIST.md`（第 19–30、61、65 行）（main `7ca692b9`） |
| 引擎分类、Pareto 口径 | `inferencex-e2e/docs/MODELS.md`（Engine submission policy；E2E normalized interactivity；North-star Pareto policy） |
| 仪表板前沿分组 | `SemiAnalysisAI/InferenceX-app` `packages/app/src/components/inference/utils/canonicalFrontier.ts` |
| 补丁 waiver | `inferencex-e2e/docs/waiver/2508.md`；InferenceX #2508、#4041 |
| 集群与钩子 | `configs/runners.yaml` 的 `mi355x-amds`；`runners/srt-slurm/hooks/{common.sh,mi355x-amds/setup.sh,check-rdma.sh,hugepage_sanity.sh}` |
| 同集群先例 | `amd-master.yaml` 的 `dsv4-fp4-mi355x-sglang-disagg-agentic-umbp-dspark`、`glm5.2-fp4-mi355x-sglang-agentic-mtp`；`kimik3/vllm/mi355x-fp4/agentx/disagg-variants.yaml`（libionic 挂载） |
| ATOM PD | [InferenceX #4045](https://github.com/SemiAnalysisAI/InferenceX/pull/4045)；sweep 39028510364 的服务端日志产物 |
| Dynamo | [InferenceX #4300](https://github.com/SemiAnalysisAI/InferenceX/pull/4300)；sglang#45111、#45311；`nvidia-master.yaml` 的 `glm5.2-fp4-gb300-dynamo-trt-agentic-disagg-mtp` |
| HiCache 撤栈 | InferenceX #4506 sweep 40201277205 的服务端日志；#2490、#2643、#3916、#4078、#4483、#4485 |
| srt-slurm | NVIDIA/srt-slurm `ae3d6877`：`src/srtctl/frontends/AGENTS.md`、`atomesh.py`；srt-slurm#433、#529；InferenceX `runners/srt-slurm/patches/README.md` |
| SGLang 补丁 | Infera `deploy/docker/patch.upstream.status.md`、`Dockerfile.sglang`；sgl-project/sglang PR（10-27 核对） |
| `write_back` 收益 | `../2p1d-aus-v0521-20261019/RESULT.zh-CN.md` §2.7 |
| Mooncake 版本 | Docker Hub 镜像构建历史；sglang `docker/rocm.Dockerfile`（v0.5.22 与 main）；kvcache-ai/Mooncake |
| Mooncake 补丁来源 | `../cross-rank-rca-20260921/REPORT.zh-CN.md`；Infera `deploy/docker/patches/mooncake_cpp/apply_mooncake_cpp_patches.sh` |
| infera overlay | Infera `deploy/overlay/{README.md,infera-exec,Dockerfile.payload}`、`manual/recipes/index.md` |
| 组件安装机制 | InferenceX `infx/srt_slurm/workload.py`（`INSTALLERS`）、`benchmarks/multi_node/srt-slurm-recipes/{RECIPES.md,configs/*.sh}`、`configs/CONFIGS.md`；srt-slurm `runtime_scripts/dynamo_wheels.py`、`core/runtime.py` |
| GitHub release 先例 | `amd-master.yaml` 的 `minimaxm3-fp8-mi300x-vllm-agentic-mtp`；`LMCache/LMCache` release `v0.5.3-rocm` |
| infera 发版 | Infera `.github/workflows/release.yml`；`AMD-AGI/Infera` release `v0.3.1` |
| 同机 PID 撞名 | `../3p1d-tp4-tp8-v0521-20261012/RESULT.zh-CN.md`（"ROCr IPC socket 撞名"） |
| 我们的数据 | `../2p1d-v0522-20261020/results/`；PR #210 描述 |
| InferenceX 数据快照 | `evidence/inferencex-api-glm52-20261027.json`（10-27 07:45 UTC） |

## 9. 复现

```bash
cd /home/liyingli/bench_agentx/baseline/Infera/llying
python3 inferencex-publish-plan-20261027/scripts/e2e_norm_table.py   # 第 3 节的两张表（我们的 7 点、ATOM PD）
```
