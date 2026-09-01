# analyze-demo 设计方案（待 review）

本文用中文写给人 review。落地时 package 内的 `*.yaml` / `readme.md` / 脚本一律用英文，遵循 `temp/mission.md` 的 RULES 第 6 条。

对应 `temp/mission.md` 第 3 阶段：**分析性能数据，产生待优化算子列表，并根据列表提供完整的 workset**。上游是 `profiling-demo` 的 `kernel_table` handoff，下游是 KernelForge。

---

## 1. 结论摘要

流水线由 5 个叶子任务组成，产出 6 种 handoff：

```
main（非叶，无 agent）
 ├── seed_table       program → kernel_table        froms: []                      ← 本阶段的 mock 输入，将来由 profiling-demo 的 kernel_scan 取代
 ├── rank             program → kernel_worklist     froms: [seed_table]
 ├── identify         program → operator_identity   froms: [rank]
 ├── build_workset   ai      → operator_workset    froms: [rank, identify]
 ├── verify_workset   program → workset_evidence    froms: [build_workset]        ← 唯一需要 GPU 的任务
 └── packup           program → analyze_packup      froms: [rank, identify, build_workset, verify_workset]  is_end
```

三个判断构成本方案的主体：

1. **6 列 Magpie CSV 不足以驱动 KernelForge**。缺的是源文件、dtype、调用点、以及可独立运行的 driver。第 4 节列出逐字段落差与补齐来源。
2. **输出格式不自造**。采用 KernelForge / Hyperloom 已有的两种格式：Hyperloom `invocation_spec` schema v2（`status: partial` + `missing_fields` 正好表达证据不完整）与 KernelForge orchestrator task YAML，两者同时产出。第 5 节展开。
3. **排序必须先分类再排序**。样例 CSV 中 NCCL 单项占 78.63%，直接 top-N 会把一个本阶段无法交给 forge-loop 的通信 kernel 排在第一。第 6 节给出分类桶与实测结果。

运行环境：`smci355-ccs-aus-n05-21`（8×MI355X / gfx950），由 Slurm 作业 28080 持有，与 `profiling-demo` 共用同一作业。本阶段**不需要模型权重，也不需要 `infera/engine-sglang:glm53-flash` 镜像**，原因见第 3.3 节。

### 实现后的修订（2026-08-31）

包已实现并在 `smci355-ccs-aus-n05-21` 上实测。四处与设计预期不符，已在代码中修正，此处记录结论，细节见 `README.md` 与各模块的 docstring。

**一、源文件解析比设计预期好得多，方法也不同。** 设计里 `min_resolve_ratio` 定在 0.6，并预判 `main_kernel` 之类必然解析失败。实测 `amd_kernel_finder` 的按名 grep 在样例数据上 5 个算子只命中 1 个，且那一个是错的（把 `add_rmsnorm_quant` 匹配到了 `triton_store_cache.py`）。改用**保持符号复合名连续、从右向左渐进缩短**的搜索后，5 个算子全部命中正确源文件。原因是设备符号里混着三类文本——mangling、算子名、tile 与调优参数——按词元切分恰好破坏了唯一有区分度的部分：`quant` 在 aiter 里到处都是。实现在 `assets/lib/symbols.py`，两条实测教训写在其 docstring 里：复合名不可从中间剥离（`a16_w16` 是算子身份不是调优参数），且要取**最左**而非最长的复合名（Triton JIT 名字里调优段比算子名更长）。

**二、handoff 内容不得包含本机绝对路径，这约束了 workset 能携带什么。** `handoff/locality.py` 在 seal 时扫描 `content/` 下每个文件，白名单只有 `/usr/`、`/opt/`、`/srv/`、`/workspace/` 等；`/apps/`、`/data/`、`/sgl-workspace/` 都在外。因此容器仓库根以 `${AITER_ROOT}` 形式传递，展开值放在包内的 `assets/lib/container_roots.yaml`。框架本有对应机制（`Oracles.image_prefixes` 接 kind 的 `dependencies`）但未接通，同事已记录为 bug 002。附带一个易踩的误报：散文里写 `<operator_id>/scripts/x.py`，其中 `/scripts/x.py` 会被判为绝对路径，因为规则的 lookbehind 不排除 `>`。为此提供了 `assets/lib/check_locality.py`，与 seal 同规则、可在 seal 之前运行。

**三、AI 任务反复超时，原因不是时长而是一个缺失的执行位。** 第一版诊断归咎于 `cli/main.py:790` 硬编码的 1800 秒静默上限，这是错的。读事件时间线才看清：完成度门禁在第 14 分钟就跑过了，只因 `items/script` 是 `-rw-rw-r--` 而拒绝——`agent/gate.py:40` 的 `EXECUTABLE_ITEMS = {script, command, entry}` 要求 `os.access(path, os.X_OK)`。交付的其余部分全部正确。问题的另一半是框架的应对：门禁失败后推送给 agent 的消息全文是 `continue, do it until finished`，不含原因，agent 无从知道要改什么，于是循环到上限。把 `top_n` 从 5 降到 2 毫无改变，这一点当时应当引起怀疑。修复后 `build_workset` 以 16 分 22 秒 succeeded，`operator_workset` 通过 seal 与 validator。本包现在两端设防：生产者设置权限位，validator 也检查它。记录在 bug 003，建议把 `GateFailure.message` 透传进 push 消息。静默上限仍是次要约束——修复后整图的四个已完成叶子占去 16 分 49 秒，留给后两叶不足 13 分钟。

**四、`verify_workset` 的容器必须以 root 运行。** 直觉上应当用 `--user` 避免留下 root 属主的文件，但 aiter 首次调用时要把 FlyDSL kernel 编译进镜像内的 `flydsl_cache/`，非 root 会以 `Permission denied` 让每个 driver 退出 1。改为保持 root 加 `PYTHONDONTWRITEBYTECODE=1`，并让 staging 清理也在 root 容器内执行——否则上一轮残留的 root 属主 `.pyc` 会让下一轮的 `rm -rf` 失败，这是只在第二次运行才出现的问题。

实测结果：两个 MoE 算子在 MI355X 上 SNR 99.49 dB 与 47.03 dB，加权平均 0.1848 ms 与 0.1058 ms，相对标准差 0.0055 与 0.0139。

### 已确认的四项决策

以下四项已经过 review 确认，本文其余章节据此写就：

| 决策项 | 结论 | 相关章节 |
|---|---|---|
| 输出格式 | **同时产出** `invocation_spec_<op>.json`（Hyperloom v2）与 `forge_task.yaml`（KernelForge task definition） | 5.1 |
| 源文件解析不出来的算子 | 本阶段标 `unknown` 并进 `missing_fields`，**不阻塞**。调用点信息由 profiling 侧后续补充，本阶段预留接收字段 | 4.4、12 |
| `verify_workset` 的容器 | 用 n05-21 已有的 `lmsysorg/sglang:v0.5.18-rocm720-mi35x`，不构建 glm53 专用镜像 | 3.3、8.5 |
| mock 输入数据 | 用现有的 GLM-5.2 decode `gap_analysis.csv`，立即开始，不等 GLM-5.3-Flash 的真实产物 | 3.1、3.2 |

---

## 2. 现场核查结果

写方案前对数据与环境做了实测，结论直接影响设计。

### 2.1 样例 CSV

对 `Infera/examples/sglang_1p1d_glm5.2/profiles/20260826_053606/decode/megapie/gap_analysis/gap_analysis.csv` 的实测：

| 项目 | 数值 |
|---|---|
| 表头 | `Name, Calls, Self CUDA total (us), Avg time (us), % Total, Input Shapes` |
| 数据行数 | 143 |
| `% Total` 求和 | 99.94 |
| `Input Shapes` 为空的行 | 13 |
| 第一名占比 | `ncclDevKernel_Generic_1` 78.63% |

前 10 名（Calls / Self CUDA total us / % Total / Name）：

```
146396  105170988.43  78.63  ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)
142878    4399200.53   3.29  main_kernel
 66150    4186038.30   3.13  mfma_moe1_silu_mul_afp4_wfp4_bf16_t32x128x256_pm1_async_v32
 66150    2314021.46   1.73  mfma_moe2_afp4_wfp4_bf16_cshuffle_t32x128x256_vscale_fix3_fp4opt_v1_persist_cu256
 60040    2203928.14   1.65  Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x32x128_MI16x16x1_...
142879     974122.85   0.73  void at::native::vectorized_elementwise_kernel<8, ...MulFunctor<float> >...
 45748     822188.34   0.61  _gemm_a16_w16_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_32_BLOCK_SIZE_K_256_...
 18522     806941.46   0.60  (anonymous namespace)::topk_transform_decode_kernel(...FastTopKParams...)
 60040     689931.43   0.52  hgemm_bf16_32x64x128x4_SPK4_W1x4x1_BLDS1_TN_AS1_0
142876     669524.98   0.50  _ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16b...
```

需要注意的是，这份数据来自 GLM-5.2 的 1P1D decode，不是 GLM-5.3-Flash。作为 mock 输入的形状样本它是合格的，但算子构成会与 GLM-5.3-Flash 不同（后者用 TileLang DSA、Triton KDA、Triton MoE）。

### 2.2 原始 trace 里有什么

对同目录下的 `1787722629.8030202-TP-0-DP-0-EP-0.trace.json.gz`（102 MB gz）实测：

| 项目 | 结果 |
|---|---|
| 事件总数 | 4,091,512 |
| 事件分类 | `cpu_op` 2,063,359 / `ac2g` 999,833 / `cuda_runtime` 696,027 / `kernel` 300,838 / `user_annotation` 15,110 / `gpu_user_annotation` 13,308 / `gpu_memcpy` 2,968 |
| **Python 调用栈** | **不存在**。事件 `args` 中没有 `Python stack` / `Call stack` 一类的键 |
| `record_shapes` | 顶层键存在，`cpu_op` 上 397,020/400,000 带 `Input Dims` / `Input type` / `Input Strides` / `Concrete Inputs` |
| `kernel_file` / `kernel_name` / `kernel_backend` | 存在于 3,541 个事件，但只对应 **5 个不同的 kernel**，且全部指向 `/tmp/torchinductor_root/...`（inductor 编译缓存，不是源码） |
| `kernel` 事件带 `External id` | 259,038 / 300,838 |

这一节有两个结论：

**dtype 是可恢复的。** `cpu_op` 上的 `Input type` 就是 TraceLens `ops_unique_args.csv` 的数据来源，通过 `External id` 与 `ac2g` 可以把 `cpu_op` 关联到设备 kernel。

**源文件不可从这份 trace 恢复。** `profiling-demo/DESIGN.md` 第 4.1 节明确要求 `with_stack:false`（理由是单 rank 文件会从 14 MB 涨到 122 MB），代价就是 Hyperloom 的 `_trace_launcher_resolver`（依赖 Python 栈帧）在这份 trace 上无法工作。热点算子里占 3.29% 的 `main_kernel` 是 TileLang 生成的通用名，既没有 `kernel_file`，也没有栈帧，仅凭 trace 无法定位。已确认这一项由 profiling 侧后续补充，本阶段只留接收字段，见第 4.4 节。

### 2.3 Magpie 自带的算子归属工具

`Magpie/tools/amd_kernel_finder/` 提供了从 profiler 符号名反查源码与测试的能力，并且已经接入 gap analysis：`Magpie/main.py:1331` 有 `--find-kernel-sources` 开关，打开后 `Magpie/modes/benchmark/gap_analysis.py:191-213` 会把 `KernelSourceInfo.csv_headers()` 追加到同一张 CSV 上。

`KernelSourceInfo`（`models.py:51`）的字段与 mission 第 3.2 节的 workset 要求高度重合：

| `KernelSourceInfo` 字段 | 对应 mission 3.2 的哪一条 |
|---|---|
| `kind`（`triton_jit` / `tensile_gemm` / `ck_tile` / `aten_native` / `hip_cpp` / `inductor` / `aiter` / `annotation` / `unknown`） | 决定 forge-loop 的 `--fellow` |
| `category`（`attention` / `gemm` / `moe_gemm` / `layernorm` / `softmax` / `copy` / `elementwise` / `indexing` / `reduce` / `router` / `kv_cache` / `blit`） | 3.2.9 算子定义格式的分类维度 |
| `source_repo` / `source_file` / `upstream_url` | 3.2.2 接入点 reference、3.2.3 截取算子本身 |
| `test_file` / `test_cmd` | 3.2.5 一键正确性/性能测试、3.2.6 测试用例 |
| `baseline_ref_file` / `baseline_ref_symbol` / `baseline_ref_kind` | 3.2.4 pytorch naive 实现 |
| `triton_ref_file` / `triton_ref_symbol` | 优化时的对照实现 |

`baseline_ref_kind` 的取值本身带语义：`eager_fn`（可 import 的纯 torch 函数）、`perftest_wrapper`（`@perftest` 包装，附带 A/B 计时）、`inline_in_test`（写死在 pytest 参数化函数体内，不可 import）、`none`。这个区分对生成 driver 很有用，因为只有前两者能直接被 driver import。

**已知缺口**：`repo_config.py` 内建的仓库只有 `rocm-libraries`、`triton`、`rocm-systems`、`aiter`、`vllm`、`pytorch`，**不含 sglang**。GLM-5.3-Flash 的 DSA / KDA / mHC 接入点都在 sglang 的 python 树里，因此需要通过 `KernelSourceFinder(repos=[...])` 显式传入 sglang 与 tilelang 的路径。

### 2.4 环境与工具链

| 项目 | 状态 | 依据 |
|---|---|---|
| Slurm 作业 28080 | 运行中，持有 `smci355-ccs-aus-n04-33` 与 `smci355-ccs-aus-n05-21` | `squeue` |
| `srun --jobid=28080 --overlap -w smci355-ccs-aus-n05-21` | **可用**，已实测执行成功 | 直接运行验证 |
| n05-21 硬件 | 8×AMD Instinct MI355X（gfx950），主机 ROCm 7.0.1 | `rocm-smi --showid` |
| n05-21 本地盘 | `/data` 49 T，可用 43 T | `df -h` |
| `/apps` 共享盘 | 52 T 已用 93%，剩 3.7 T | `df -h`，大产物不要写这里 |
| n05-21 上的 sglang 镜像 | 有 `lmsysorg/sglang:v0.5.18-rocm720-mi35x`（正是 glm53flash-demo 的 base image） | `docker images` |
| n05-21 上的 `infera/engine-sglang:glm53-flash` | **没有** | `docker images` |
| n05-21 `/data/models` | 只有 `Kimi-K3`，无 GLM 权重 | `ls` |
| `agent-sys` CLI | **未安装** | `which agent-sys` 为空 |
| TraceLens | 源码在 `/apps/.../TraceLens`，**未安装**（`import tracelens` 失败） | 实测 |
| SIKL 仓库 | **本地不存在**，`/apps` 下无任何 SIKL 目录 | 实测 |

SIKL 找不到这件事影响 mission 3.2.9「算子定义格式参考 SIKL」。第 2.5 节给出替代来源，第 15 节列为待确认问题。

补充核查结论：`AMD-AGI` 组织本身可读（68 个 public repo，含 `Infera` / `Hyperloom` / `Magpie` / `TraceLens` / `AgentKernelArena`），但 `AMD-AGI/SIKL` 在未认证条件下返回 HTTP 404，说明它是该组织下的 **private 仓库**。mission.md 第 42 行本身也写了「请联系 xiaobo 或 huangzhen」，与这个判断一致。本机 `/apps/xiaobo` 存在但只有 sglang 启动脚本，无 SIKL 副本。

### 2.5 AgentKernelArena 是本阶段输出的现成参照

`/apps/.../AgentKernelArena/tasks/image_kernel/` 下有 **21 个任务，全部是 Hyperloom session 的忠实复现**。它们是"分析阶段产出什么"这个问题的既有答案，而且是本地可读的。目录命名规则是 `<gpu>_<framework>_<language>_<operator>`，例如：

```
mi355x_vllm_tilelang_mhc_fused_post_pre          TileLang
mi355x_vllm_triton_kda_linear_attn_kimi_k3       Triton，KDA 线性注意力
mi355x_vllm_triton_fused_moe_gemma4              Triton MoE
mi355x_sglang_triton_mxfp8_grouped_gemm          sglang + Triton
mi300x_sglang_hip_pa_decode                      sglang + HIP
```

其中 TileLang、KDA、Triton MoE 三类正是 GLM-5.3-Flash 的算子构成（`--dsa-*-backend tilelang`、KDA 层、`--moe-runner-backend triton`），因此这些样例不只是格式参照，内容上也直接相关。

每个任务由四部分组成：

| 文件 | 内容 |
|---|---|
| `config.yaml` | 算子身份、容器内仓库路径、可编辑源文件、目标函数、三个命令、超时、平台门槛、给 agent 的自然语言简报 |
| `session_cases.json` | 从 Hyperloom session 解析出的 workload case：`gpu_pct` / `kernel_ids` / `params` / `trace_input_shapes` / `benchmark` / `seed` |
| `scripts/task_runner.py` | compile / correctness / performance 三个子命令的统一入口 |
| `scripts/{standalone_driver,forge_driver}.py` | **两个 driver**：一个独立运行，一个满足 forge-loop 的 stdout 契约 |

`session_cases.json` 的 `selection` 字段记录的是筛选算法本身。`mi355x_vllm_triton_unified_attention` 那份写的是：

```
"selection": "routable_kernels; gpu_pct>=3; session-derived shapes/dtypes"
```

这与第 6 节设计的"先分 routable 桶、再按占比过滤、shape 从 trace 证据来"完全一致，是一个独立的印证。

---

## 3. 输入契约

### 3.1 mock 的做法：`seed_table` 任务

agent_sys 的 handoff 只能由同一张图里的上游任务产生，没有"从外部注入一个 handoff"的入口。因此 mock 的形态是**图头部的一个 program 叶子任务** `seed_table`，`froms: []`，它做的事只有一件：把包变量 `seed_csv` 指向的 CSV 复制成一个 `kernel_table` handoff，并解析出 `text.json`。

这样做有两个好处。一是 `kernel_table` 这个 kind 的定义与 `profiling-demo/DESIGN.md` 第 4.2 节里 `kernel_scan` 产出的 kind 完全一致（同名、同 `content_type`、同 items），将来合图时把 `seed_table` 从 subgraph 里删掉、把 `rank` 的 `froms` 改成 `[kernel_scan]` 即可，其余不动。二是 mock 与真实输入走同一条校验路径，`check_kernel_table` validator 两边共用。

`kernel_table` kind 的定义（与 profiling-demo 对齐）：

| 项目 | 值 |
|---|---|
| `content_type` | `structured_text` |
| items | `text.json`（top-N kernel 的结构化列表）、`gap_analysis/`（Magpie 原始 CSV 目录） |
| README 必需小节 | Purpose / Schema |

### 3.2 输入 CSV 的两种形态

`seed_table` 接受两种 CSV，通过表头列数区分：

- **6 列基础形态**：`Name, Calls, Self CUDA total (us), Avg time (us), % Total, Input Shapes`。这是当前样例数据的形态，也是 `profiling-demo` 当前设计的产出。
- **19 列增强形态**：基础 6 列 + `KernelSourceInfo.csv_headers()` 的 13 列。这是 Magpie 加 `--find-kernel-sources` 之后的产出。

在输入是增强形态的情况下，`identify` 任务可以直接读取已有列，跳过重新索引仓库的开销；在输入是基础形态的情况下，`identify` 自己调用 `KernelSourceFinder` 补齐。两条路径产出同一个 `operator_identity` handoff。

### 3.3 为什么本阶段不需要模型权重和自定义镜像

已确认 `verify_workset` 使用 n05-21 上现成的 `lmsysorg/sglang:v0.5.18-rocm720-mi35x`，不构建 `infera/engine-sglang:glm53-flash`。

需要 GPU 的只有 `verify_workset`，它做的是**单算子的独立正确性与性能测试**，不启动推理服务。算子实现本身（aiter、triton、tilelang、torch）都在 `lmsysorg/sglang:v0.5.18-rocm720-mi35x` 这个 base image 里，n05-21 上已有。

GLM-5.3-Flash 特有的部分是 sglang PR #36507 的 python overlay（`glm5_next.py` 等模型代码），它提供的是**接入点信息**——某个 kernel 在模型里被谁调用、以什么形状调用。这部分是"读"而不是"运行"，从 git 检出源码即可，不必构建 306 GB 权重能加载的镜像。

由此，`profiling-demo` 第 7 节列的三项前置准备（构建镜像、准备权重、拉 AIPerf 镜像）对 analyze-demo 全部不适用。

---

## 4. 落差分析：从 Magpie CSV 到 KernelForge

这一节回答用户问题里的"这一阶段需要提供给 KernelForge 哪些信息"。

### 4.1 KernelForge 的两条输入路径

KernelForge 有两个入口，输入契约不同，需要先明确本阶段对哪一个：

| 路径 | 命令 | 输入形态 |
|---|---|---|
| Orchestrator（多 fellow campaign） | `kernel-agents run tasks/<name>.yaml` | 一个 YAML，格式见 `docs/reference/task-definition.md` |
| Autonomous loop（Hyperloom 实际调用的） | `kernel-agents forge-loop --kernel ... --driver ...` | 一个 git workspace + 满足 stdout 契约的 `driver.py`，可选 `--invocation-spec-file` |

Hyperloom 的 `agents/kernel/tools/backends/forge_submit.py` 走的是第二条。它传给 forge-loop 的完整参数集是：`--kernel` `--driver` `--workspace` `--snr-threshold` `--max-iters` `--max-hours` `--git-branch` `--gpu-target` `--gpu-type` `--fellow` `--experiments-dir` `--experiment-id` `--experience-id` `--deadline-unix` `--result-json`，以及条件附加的 `--program-md-file` `--invocation-spec-file` `--operator-name` `--target-functions` `--source-files` `--framework`。

mission 第 4 节写的是"接入 kernel forge（嵌套 ai agent backend）… 输入包含 workset 的完成"，对应的是第二条路径。因此本阶段的 workset 要能拼出上面这组参数。

### 4.2 逐字段落差表

| KernelForge / forge-loop 需要 | 6 列 CSV 是否具备 | 补齐来源 | 本阶段谁负责 |
|---|---|---|---|
| `--kernel`（被编辑的源文件） | 否，只有设备符号名 | `amd_kernel_finder` 的 `source_file` | `identify` |
| `--source-files`（多文件算子） | 否 | 同上 + `kernel_sources` 推导 | `identify` |
| `--target-functions`（PMC 过滤用符号） | 否 | 从 `source_file` 解析符号 | `identify` |
| `--fellow`（triton / ck / flydsl / hip） | 可从符号名推断 | `KernelKind` | `identify` |
| `--operator-name`（逻辑算子名） | 可从符号名推断 | `KernelCategory` + 规范化 | `identify` |
| `--framework` | 否 | 包变量固定 `sglang` | `rank` |
| `--gpu-target` / `--gpu-type` | 否 | `glm53flash-demo/environment.md`：`gfx950` / `mi355x` | `rank`（写进 env 快照） |
| `--snr-threshold` | 否 | 包变量，默认 30.0 | `rank` |
| `--driver`（stdout 契约） | 否 | 由 `test_file` / `test_cmd` / `baseline_ref_*` 生成 | `build_workset` |
| `--program-md-file` | 否 | 生成 | `build_workset` |
| `invocation.arguments[].shape` | 是（字符串形态） | 解析 `Input Shapes` | `rank` |
| `invocation.arguments[].dtype` | 否 | trace 的 `cpu_op.Input type`；或从符号名推断（`afp4_wfp4_bf16` 这类命名已编码了 dtype） | `identify` |
| `invocation.launcher_locator`（调用点） | 否 | 需要 `with_stack:true` 的 trace，**当前不具备** | 留空，标 `missing_fields` |
| `edit_target.repo_root` | 否 | 容器内 sglang / aiter 路径，包变量 | `identify` |
| `tests.driver_contract.case_selectors` | 部分 | `Input Shapes` 去重 → `case_001..N` | `rank` |
| `workload.call_count` | 是 | `Calls` | `rank` |
| `deployment.batch` / `deployment.sequence` | 否 | 来自 profiling 阶段的 `aiperf_report`（并发、请求长度） | 缺失，见第 12 节 |
| 性能基线 wall time | 部分 | `Avg time (us)` 是**服务内平均**，不是独立 bench 的 wall time，两者不可直接比较 | `verify_workset` 重新实测 |
| pytorch naive 参考实现 | 否 | `baseline_ref_file` / `baseline_ref_symbol` / `baseline_ref_kind` | `identify` + `build_workset` |
| 3 个以上正确性用例 | 否 | shapes 去重后取 3 个以上，配 SNR / allclose 判据 | `build_workset` |

### 4.3 forge-loop 的 driver stdout 契约

这是 workset 里技术含量最高的一件产物，格式是硬要求（`KernelForge/src/kernel_agents/loop/task_preparer.py` 的 `DRIVER_CONTRACT_SPEC`）：

```
python driver.py                                        跑完整正确性套件，至少打印一行
    SNR: 62.13 dB                                       首选，forge 用它与阈值比较
    allclose: True                                      备选
python driver.py --warmup <n> --iters <n> --bench-mode  打印基准耗时
python driver.py --profile-run                          单次前向，供 profiler 采样
```

多 case 时通过 `--shape` 与 `CASE_ID` 选择器传入，driver 必须覆盖 `case_selectors` 里声明的全部 case，preflight 阶段会拒绝缺 case 或多 case 的 driver。

`driver.py` 在 forge-loop 中是**受保护文件**，agent 不允许修改它。这意味着 driver 的正确性由本阶段独立担保——这正是 `verify_workset` 存在的理由。

### 4.4 workset 的单位是框架层入口，不是设备 kernel 符号

这一节的判断在读了 AgentKernelArena 的 21 个既有任务之后做了修正，是本文与初稿差异最大的一处。

初稿把"解析不出 `main_kernel` 的源文件"当成一个需要容忍的缺陷。实际上 Arena 的做法说明这不是缺陷，而是**问题提错了层次**。以 `mi355x_vllm_triton_kda_linear_attn_kimi_k3` 为例，它的 `selection` 字段直接写明了同类情况：

```
KDA is Triton-JIT so the trace shows launcher = Not found;
the entry points below were recovered from the session call stacks.
```

它没有去解析 Triton JIT 生成的设备符号，而是把 `target_kernel_functions` 定位到框架层的两个 Python 可调用入口 `fused_recurrent_kda_packed_decode` 与 `chunk_kda_with_fused_gate`，`source_file_path` 列出 9 个相关文件。TileLang 的样例同理：`mi355x_vllm_tilelang_mhc_fused_post_pre` 的 `source_file_path` 是 `tilelang.py`，`target_kernel_functions` 是 `mhc_fused_post_pre_tilelang`，完全不涉及 TileLang 生成的设备符号名。

出现这个现象的原因是有以下两点。第一，JIT 类后端（Triton / TileLang）的设备符号是编译产物，改它没有意义，可编辑的对象本来就是生成它的 Python 函数。第二，forge-loop 的 `--kernel` 参数要的是"被编辑的源文件"，它天然就是框架层文件而不是设备符号。

因此 `identify` 任务的目标改为：**从设备 kernel 符号出发，向上定位到框架层的可调用入口**。产出的 `operator_identity` 字段相应对齐 Arena 的 `config.yaml`：

| 字段 | 含义 | 对应 forge-loop 参数 |
|---|---|---|
| `image_repo_path` | 容器内仓库根，例如 `/sgl-workspace/sglang/python/sglang` | `--workspace` 的组成 |
| `repo_subdir` | 仓库子目录名 | — |
| `repository_language` | `triton` / `tilelang` / `hip` / `ck` / `aiter` | `--fellow` |
| `source_file_path[]` | 相关源文件，可多个 | `--source-files` |
| `editable_sources[]` | 其中真正允许被编辑的 | `--kernel` |
| `target_kernel_functions[]` | 框架层入口函数名 | `--target-functions` |
| `kernel_identity.logical_operator` | 逻辑算子名 | `--operator-name` |
| `kernel_identity.kernel_kind` | 同 `repository_language` | — |
| `kernel_identity.source_owner` | `sglang` / `aiter` / `vllm` | `--framework` |

`amd_kernel_finder` 仍然有用，但它的定位从"唯一解析手段"降为"第二级线索"：它按符号名反查出的 `source_file` 用来指示**方向**（哪个仓库、哪个子系统），最终的 `target_kernel_functions` 由 `build_workset` 的 AI 任务结合 sglang 源码确定。这也是 Arena 那句 "recovered from the session call stacks" 描述的人工或 agent 参与的环节。

调用点信息（`invocation.launcher_locator`）在 profiling 侧补齐之后会让这一步更省事，但**不是前置条件**。第 2.2 节确认当前 trace 没有 Python 调用栈，已确认的处理方式是：本阶段不为此做额外工作，也不因此阻塞。

为了让那批信息到位时不必改动本包的结构，`kernel_table` kind 的 `text.json` 里预留一个可选块：

```json
{
  "kernels": [
    {
      "name": "...",
      "calls": 66150,
      "self_us": 4186038.30,
      "pct_total": 3.13,
      "input_shapes": "...",

      "launcher": {                      // 可选。profiling 侧补齐后出现
        "source_file": "/sgl-workspace/sglang/python/sglang/srt/layers/moe/...",
        "line": 412,
        "function": "fused_moe_forward",
        "launch_api": "hipModuleLaunchKernel",
        "sample_count": 128
      }
    }
  ]
}
```

字段命名对齐 Hyperloom `_trace_launcher_resolver.py` 的 `LauncherFrame`（`source_file` / `line` / `function` / `sample_count` / `launch_api`），这样两边不需要转换层。

`identify` 任务的解析顺序因此是三级降级：

1. `kernel_table` 里带 `launcher` 块 → 直接得到框架层文件与函数名，`source_resolution_method: trace_python_stack`
2. 没有 `launcher` 块 → 调 `amd_kernel_finder` 按符号名反查得到仓库与子系统方向，`source_resolution_method: name_grep`
3. 两者都只给出方向而定不到具体入口函数 → 记录 `resolution_hint`（仓库路径 + 候选目录 + 符号名特征），交由 `build_workset` 在源码中确定 `target_kernel_functions`，`source_resolution_method: agent_recovered`

第三级不判失败，它是 Arena 里 KDA 与 TileLang 两个样例实际走的路径。`check_identity_resolved` 的 `min_resolve_ratio = 0.6` 约束的是前两级的合计比例，第三级的算子在 `check_workset_shape` 处才被检查——那时 `target_kernel_functions` 必须非空且能在 `source_file_path` 里定位到。这样把"符号名解析不出来"与"入口函数最终没找到"区分成两件事，前者是常态，后者才是问题。

---

## 5. 输出格式选型

### 5.1 备选与推荐

mission 3.2.9 说算子定义格式参考 SIKL，但 SIKL 在本地不存在（第 2.4 节）。在无法获取 SIKL 的情况下，有三个选项：

**选项 A：Hyperloom `invocation_spec` schema v2。** 由 `Hyperloom/src/hyperloom/agents/kernel/tools/_invocation_spec.py` 定义，12 个顶层块：`schema_version` / `status` / `missing_fields` / `logical_operator` / `source_framework` / `implementation` / `kernel` / `edit_target` / `invocation` / `tests` / `workload` / `execution` / `deployment` / `provenance`。优点是 forge-loop 的 `--invocation-spec-file` 直接接受它，且它自带 `status: "complete" | "partial"` 与 `missing_fields` 列表，可以诚实表达"我们只有部分证据"。缺点是字段多，其中 `launcher_locator`、`runtime_symbols`、`unresolved_runtime_symbol_prefixes` 依赖本阶段拿不到的数据。

**选项 B：KernelForge orchestrator task YAML。** 由 `KernelForge/docs/reference/task-definition.md` 定义：`task_id` / `description` / `operation` / `dtype` / `gpu_target` / `shapes.primary` / `shapes.validation` / `backends` / `targets.{snr_db,wall_ms,baseline_wall_ms}` / `constraints` / `phases`。优点是字段少、人可读、可 review；KernelForge 的 sglang fellow 自己产出的就是这个格式的扩展（多了 `kernels_to_review[].measured_baseline` 与 `infrastructure`）。缺点是它对应 campaign 路径，不是 Hyperloom 走的 forge-loop 路径。

**选项 C（读了 Arena 之后新增）：AgentKernelArena `config.yaml` + `session_cases.json`。** 这是本地唯一有 21 个真实样例背书的格式，且它就是 Hyperloom 分析阶段产物的落地形态。它覆盖了 A 与 B 都没有的三类信息：容器内仓库路径与可编辑文件（`image_repo_path` / `editable_sources`）、分项超时与平台门槛（`compile_timeout` / `platform_support.required_arch`）、以及给 agent 的自然语言简报（`prompt.instructions`）。在 SIKL 拿不到的前提下，**它是最接近 mission 3.2.9 所要求的"标准算子定义格式"的现成物**。

**选项 D：自定义 schema + 双向适配器。** 不推荐，理由是下游三个消费者都已经有既定格式，新造一层只增加维护面。

**已确认采用 A 与 B 同时产出。** 两者在信息量上不冲突：invocation_spec 是机器契约，直接喂给 `forge-loop --invocation-spec-file`；task YAML 是人可读的意图声明与 review 载体，也是 KernelForge sglang fellow 自己产出的格式。同一份 `operator_identity` 数据生成两份文件的成本很低。工程上的表述是：`operator_workset` handoff 内每个算子目录同时含 `invocation_spec_<op>.json` 与 `forge_task.yaml`。

两份文件的字段映射由 `assets/lib/forge_export.py` 一处实现，`check_workset_shape` 校验两者的公共字段一致（`gpu_target`、primary shape、`snr_db` 与 `snr_threshold`），避免两份文件各说各话：

| 概念 | `invocation_spec_<op>.json` | `forge_task.yaml` |
|---|---|---|
| 算子标识 | `logical_operator`、`kernel.name` | `task_id`、`operation` |
| 硬件 | `execution.target_platform` | `gpu_target` |
| 主形状 | `workload.task_group.cases[0]` | `shapes.primary` |
| 校验形状 | `workload.task_group.cases[1:]` | `shapes.validation` |
| 正确性门槛 | `tests.driver_contract` + CLI `--snr-threshold` | `targets.snr_db` |
| 性能基线 | 无对应字段，由 `workset_evidence` 承载 | `targets.baseline_wall_ms` |
| 源文件 | `edit_target.source_file`、`implementation.sources` | `paths.reference`、`source_files` |
| 证据缺口 | `status` + `missing_fields` | `constraints` 里以 `TODO:` 前缀列出 |

**建议把选项 C 加为第三份产出，需要 review。** 理由是它是唯一携带容器仓库路径、可编辑文件白名单与分项超时的格式，而这三项恰好是 `verify_workset` 与下游 forge-loop 都需要的运行时事实；A 与 B 都表达不了。增量成本很低：`config.yaml` 的字段全部来自已有的 `operator_identity`，`session_cases.json` 的 `cases[]` 与 A 的 `workload.task_group.cases` 是同一份数据的另一种排布。这一项列在第 15 节。

### 5.2 六个 handoff kind

| kind | content_type | items | 说明 |
|---|---|---|---|
| `kernel_table` | `structured_text` | `text.json`、`gap_analysis/` | 与 profiling-demo 同名同构，本阶段由 `seed_table` mock |
| `kernel_worklist` | `structured_text` | `text.json`、`schema`、`worklist.csv` | **待优化算子列表本身**。含全部 143 行的分类结果与排除理由，以及选中的 top-N |
| `operator_identity` | `structured_text` | `text.json`、`schema` | 每个选中算子的源码归属：`source_file` / `test_file` / `baseline_ref_*` / `kind` / `category` / dtype |
| `operator_workset` | `reproducible` | `result`、`env`、`script`、`code`、`logs`、`watchout` | 每算子一个子目录，内含交给 KernelForge 的全套材料 |
| `workset_evidence` | `structured_text` | `text.json`、`logs` | `verify_workset` 的实测结果：每个算子的正确性与性能数字 |
| `analyze_packup` | `reproducible` | 按 `experiment-result-packup` 的 `deliverable_layout.md` 组织 | 交付物 |

`reproducible` 强制要求 `result` 与 `env` 两个 item，`script` / `command` 至少有一个，README 必须含 Purpose / How to run / Result / Environment / Watch out 五节。`structured_text` 只要求 `text.json` / `text.yaml` / `text.xml` 三选一，README 需含 Purpose / Schema。

### 5.3 `operator_workset` 的目录形态

每个选中算子一个目录，对照 mission 3.2 的九条要求：

```
items/code/<operator_id>/
├── README.md                       算子说明、来源、为什么选它
├── invocation_spec_<op>.json       选项 A：Hyperloom schema v2
├── forge_task.yaml                 选项 B：KernelForge task definition
├── config.yaml                     选项 C：Arena image_kernel 格式（若采纳第 15 节问题 1）
├── session_cases.json              选项 C 的 workload case（若采纳）
├── program.md                      给 forge fellow 的任务简报，内容对应 Arena 的 prompt.instructions
├── run_forge.sh                    一键拼出 forge-loop 命令行
├── kernel/
│   ├── source_ref.md               3.2.3 截取算子本身：源文件路径、行范围、upstream URL、commit
│   └── snapshot/                   源文件快照（按需，见第 12 节体量取舍）
├── reference/
│   └── naive_torch.py              3.2.4 pytorch naive 实现（来自 baseline_ref_* 或生成）
├── scripts/
│   ├── task_runner.py              compile / correctness / performance 三个子命令的统一入口
│   ├── standalone_driver.py        独立运行，给人用
│   └── forge_driver.py             3.2.5 满足 forge-loop stdout 契约，给 forge-loop 用
├── graph_harness.py                HIP graph 计时 harness，从 KernelForge examples 原样复制
├── tests/
│   └── cases.json                  3.2.6 至少 3 个正确性用例（shape / dtype / 容差）
├── integration/
│   └── sglang_hookpoints.md        3.2.2 sglang 接入点 reference 与说明
└── provenance.json                 3.2.8 截取时 profile 报告中的性能（Calls / Avg time / % Total / trace 路径）
```

**两个 driver 而不是一个**，这一点从 Arena 移植而来。`forge_driver.py` 受 stdout 契约约束（必须打印 `SNR: <x> dB`、接受 `--bench-mode` / `--profile-run`），格式刚性；`standalone_driver.py` 面向人，可以打印表格、对比多个实现、接受任意参数。两者共享 `task_runner.py` 里的算子调用与数据构造逻辑，避免两处实现漂移。`verify_workset` 调 `task_runner.py`，forge-loop 调 `forge_driver.py`。

`reference/naive_torch.py` 遵循 KernelBench 约定（`class Model(nn.Module)` + `get_inputs()` + `get_init_inputs()`），这是 Arena 的 `torch2*` 类任务使用的形态，张量在 CPU 上构造、由 harness 负责搬运。KernelForge 本身不使用这个约定，但它是本地唯一有样例的 PyTorch 参考实现规范，对应 mission 3.2.4。

3.2.1 实验环境与 3.2.7 性能测试结果不放在每个算子目录里，而是分别放在 `items/env/`（全包共享一份环境快照）与 `workset_evidence` handoff（实测数据独立成 handoff，便于 validator 单独校验）。

### 5.4 本阶段生产什么，不生产什么

一句话概括：**本阶段生产的是测量装置，不是被测对象。**

**不生产 kernel 源码。** 被优化的 kernel 已经存在于框架里——sglang 的 python 树、aiter、TileLang 后端。forge-loop 的 `--kernel` 参数要的是一个**位于 git workspace 中的真实文件路径**，因为它要在那里做编辑、提交、diff、导出 patch。给它一份 handoff 里的拷贝没有意义。本阶段对源码做的事只有定位：产出 `image_repo_path`（容器内仓库根）、`source_file_path[]`（相关文件）、`editable_sources[]`（其中允许被编辑的）、`target_kernel_functions[]`（入口函数名）。这与 Arena 的 `image_kernel` 任务类型完全一致——那 21 个任务的目录里都没有 kernel 源码，只有指向容器内路径的声明。

`kernel/snapshot/` 是可选项，用途是留证与 review（记录"截取时这个文件长什么样"），不是给 KernelForge 读的。它受 `max_snapshot_kb` 限制，超了就退化成只记路径与 commit。

**生产单元测试，而且这是本阶段最要紧的产出。** forge-loop 把 `--driver` 指向的文件当作**受保护文件**，优化 agent 不允许修改它。也就是说，判定"改完之后还对不对、快了多少"的全部依据，来自本阶段写的这份 driver。它写错了，forge-loop 会朝着错误的方向优化并给出看起来合理的报告。这也是 `verify_workset` 必须存在的理由：在交给 KernelForge 之前，先在 GPU 上把 driver 实际执行一遍，确认它能打印出 SNR 行、数值达标、5 组 bench 的相对标准差在容忍范围内。

按 forge-loop 的输入逐项对照：

| forge-loop 的输入 | 本阶段是否生产 | 说明 |
|---|---|---|
| `--kernel` 指向的源文件 | **否**，只定位 | 框架里已有，forge-loop 在 git worktree 里就地编辑 |
| `--source-files` | **否**，只定位 | 同上 |
| `--driver` | **是** | 正确性判据 + 性能测量，受保护文件，本阶段独立担保 |
| `graph_harness.py` | 复制 | 从 `KernelForge/examples/` 原样取，不自己写计时逻辑 |
| `--program-md-file` | **是** | 自然语言简报，对应 Arena 的 `prompt.instructions` |
| `--invocation-spec-file` | **是** | 形状、dtype、case 选择器、算子身份 |
| `forge_task.yaml` | **是** | 人可读的意图声明与验收门槛 |
| `reference/naive_torch.py` | **是**（或 import 现成的） | 正确性的真值来源 |
| `cases.json` | **是** | 3 个以上正确性用例 |

`reference/naive_torch.py` 有一个例外：在 `operator_identity` 的 `baseline_ref_kind` 为 `eager_fn` 或 `perftest_wrapper` 时，说明 aiter 或 vllm 里已经有可 import 的纯 torch 实现，这时应当直接 import 而不是重写。重写一份等价实现是引入错误的常见来源，而正确性判据的可信度是这一步的全部价值。

---

## 6. 排序与筛选算法

mission 说"待优化算子列表的产生算法暂时不知重点，存在即可"，因此这里只求可解释、可复现，不追求最优。但**分类环节不能省**，理由在第 1 节已述。

### 6.1 五个分类桶

按符号名模式分类，规则表用 YAML 写在 `assets/lib/kernel_taxonomy.yaml` 里，便于 review 与增补：

| 桶 | 匹配模式 | 是否进入候选 | 理由 |
|---|---|---|---|
| `collective` | `ncclDevKernel*` / `rccl*` | 否 | forge-loop 的常规 driver 无法测单卡通信 kernel；Hyperloom 对它另有 `collective_driver_generator.py` 路径 |
| `vendor_tuned` | `Cijk_*`（Tensile/rocBLAS 汇编生成）、`hgemm_bf16_*` | 否 | 优化手段是调 tuning 表而不是改源码，与 forge-loop 的编辑循环不匹配 |
| `framework_native` | `at::native::*` | 否 | PyTorch ATen 通用 kernel，改动影响面超出本任务范围 |
| `routable` | `_ZN5aiter*`、`mfma_*`、`_gemm_*_kernel_*`（Triton）、`main_kernel`（TileLang）、`*sgl*`、其余带明确 owner 的 | **是** | 有源文件、可独立构建与测量 |
| `unknown` | 以上都不匹配 | 否，但记录 | 供人工 review，是规则表的增补来源 |

分类结果全部写进 `kernel_worklist`，被排除的行带 `excluded_reason` 字段，不静默丢弃。

### 6.2 排序与截断

`routable` 桶内按 `% Total` 降序，加两个下限过滤：`min_pct`（默认 0.1）与 `min_calls`（默认 100），再取前 `top_n`（默认 5）。

对样例 CSV 实测，`routable` 桶的前 5 名会是：

| 排名 | Name | % Total | Calls | 预期 `kind` |
|---|---|---|---|---|
| 1 | `main_kernel` | 3.29 | 142878 | TileLang，名字不带信息，是 `identify` 的难点 |
| 2 | `mfma_moe1_silu_mul_afp4_wfp4_bf16_t32x128x256_pm1_async_v32` | 3.13 | 66150 | aiter / CK asm，MoE GEMM |
| 3 | `mfma_moe2_afp4_wfp4_bf16_cshuffle_t32x128x256_vscale_fix3_fp4opt_v1_persist_cu256` | 1.73 | 66150 | 同上 |
| 4 | `_gemm_a16_w16_kernel_BLOCK_SIZE_M_32_...` | 0.61 | 45748 | Triton，参数已编码在名字里 |
| 5 | `(anonymous namespace)::topk_transform_decode_kernel(...)` | 0.60 | 18522 | sgl-kernel HIP |

被排除的 NCCL 虽然占 78.63%，但会在 `kernel_worklist` 里以 `bucket: collective, excluded_reason: not_routable_by_forge_loop` 出现，并在 README 中说明"通信开销是本次 profile 的主要构成，属于并行策略问题而非单算子问题"。这个信息对 e2e 优化本身是有价值的，只是不属于本阶段的交付路径。

### 6.3 shape 去重与 case 生成

`Input Shapes` 列是分号分隔的多组形状，例如：

```
[288,6144]x[33,4096,3072]x[33,6144,1024]x[288,9]x[288,9]x[257]x[33,4096,192]x[33,6144,64]; [256,6144]x...
```

解析规则是：`;` 分组，`x` 分参数，`[a,b,c]` 是一个张量的维度。去重后取前 `max_cases`（默认 4）组，第一组作为 `primary`，其余作为 `validation`，编号 `case_001` 起。这与 Hyperloom `_task_group_contract.py` 的 `build_task_group_shape_cases` 行为一致（`CASE_SELECTOR_KEY = "CASE_ID"`）。

13 行 `Input Shapes` 为空的算子无法生成 case，`rank` 会给它们打 `no_shape_evidence` 标记并排除。

---

## 7. agent_sys 图设计

### 7.1 为什么是 5 个叶子（加 1 个 mock）

拆分的依据是**失败模式与成本量级不同**：

- `seed_table` 秒级、纯本地文件操作。它是将来要删掉的一块，独立成任务才能干净地删。
- `rank` 秒级、纯 CSV 计算、无网络无 GPU。它的失败只可能是解析问题。
- `identify` 分钟到十分钟级、**需要网络**（`amd_kernel_finder` 会 clone aiter / vllm / pytorch / triton / rocm-libraries）、无 GPU。它的失败是仓库不可达或符号匹配不上。
- `build_workset` 是**唯一的 AI 任务**，分钟级、需要 Anthropic endpoint、无 GPU。生成 driver 与 naive 实现属于代码生成，程序化模板做不了。
- `verify_workset` 分钟到小时级、**需要 8×MI355X**、需要容器。它的失败是编译或数值问题。
- `packup` 秒级。

把 `identify` 与 `build_workset` 合并会让"仓库 clone 失败"与"模型生成质量不足"两类问题混在一次 attempt 里，重试代价与定位难度都上升。把 `verify_workset` 并入 `build_workset` 会让 AI 任务持有 GPU，与 mission"稳定可复现"的目标相悖。

### 7.2 为什么 `build_workset` 用 AI 后端

`profiling-demo/DESIGN.md` 第 11 节的开放问题 5 提到"本轮是否需要至少一个 AI task……更贴合 analyze 阶段的算子清单生成"。这里给出确认：AI 任务放在 `build_workset`，不放在 `rank`。

原因是有以下几点。第一，`rank` 是确定性计算，用模型只会引入不可复现性。第二，`build_workset` 要产出 `driver.py`（满足 stdout 契约）、`naive_torch.py`（算子的 PyTorch 等价实现）、`program.md`（给 fellow 的自然语言简报），三者都是代码或散文生成，模板法只能覆盖最规整的 GEMM 类算子。第三，mission 的 ISSUES 第 2 条要求指定 agent backend 为 claude code sdk，需要至少一个 AI 任务来验证这条链路。

agent 定义（`kind: ai`，`backends: [{key: claude_code_sdk, backend_entry: agent.backends.claude_sdk:ClaudeSdkBackend}]`）从 `demo/steps/describe.yaml` 移植。包变量通过 agent 的 `env:` 块传入，这是 AI 任务读取变量的唯一途径。

### 7.3 六个 validator

全部是 program validator，本阶段不引入 AI validator。`validator/report.py` 的 `blocks_the_task` 规则要求每个 handoff kind 至少绑定一个 validator，否则判 `unchecked` 并阻塞，下表已满足。

| validator | dimension / strength | 挂在 | 检查内容 |
|---|---|---|---|
| `check_kernel_table` | usability / strong | `kernel_table` | 从 profiling-demo 移植：表头是 6 列或 19 列；数据行数 ≥ `min_kernel_rows`（默认 20）；`% Total` 求和落在 (0, 100] |
| `check_worklist_shape` | completeness / strong | `kernel_worklist` | 每一行都有 `bucket`；被排除行都有 `excluded_reason`；选中数量在 [1, `top_n`]；选中行的 `% Total` 单调不增；分类桶覆盖率（`unknown` 占比低于 `max_unknown_ratio`，默认 0.3） |
| `check_identity_resolved` | trustworthiness / strong | `operator_identity` | 每个选中算子有非空 `kind` 与 `category`；`source_file` 解析率不低于 `min_resolve_ratio`（默认 0.6）；解析出的路径在声明的 repo_root 下真实存在 |
| `check_workset_shape` | completeness / strong | `operator_workset` | 每个算子目录含第 5.3 节列出的必需文件；`cases.json` 至少 3 个 case；`invocation_spec_*.json` 能被 `json.load` 且 `schema_version == 2`；`forge_task.yaml` 含 `task_id` / `gpu_target` / `shapes.primary` / `targets.snr_db` |
| `check_workset_runs` | trustworthiness / strong | `workset_evidence` | 每个算子的 `python driver.py` 打印了 SNR 或 allclose 行；SNR ≥ `snr_threshold`（默认 30.0）；bench 有 5 组结果且每组 loop ≥ 10（对应 mission 3.2.7）；相对标准差低于 `max_rsd`（默认 0.1） |
| `check_packup_shape` | completeness / strong | `analyze_packup` | 从 `single_real_task/assets/check_packup_shape.validator` 移植 |

`check_identity_resolved` 的 `min_resolve_ratio` 默认取 0.6 而不是 1.0，是因为第 2.2 节已经确认 `main_kernel` 这类 TileLang 生成名在没有 Python 栈的前提下大概率解析不出来。把门槛设成 1.0 会让任务在一个已知且已记录的限制上失败。

---

## 8. 各任务实现要点

### 8.1 `seed_table`

读 `--var seed_csv` 指向的 CSV，校验表头，复制到 `$AGENT_SYS_OUTPUT_KERNEL_TABLE/items/gap_analysis/`，同时解析成 `text.json`。默认值指向样例数据的绝对路径，不把 66 KB 的 CSV 放进包里，避免每个 attempt zone 都复制一份。

README 里要写明这是 mock：数据来自 GLM-5.2 1P1D decode，不是 GLM-5.3-Flash，形状可用于流程验证但不代表目标模型的算子构成。

### 8.2 `rank`

纯 Python，无外部依赖。三步：按 `assets/lib/kernel_taxonomy.yaml` 分类 → 过滤排序截断 → 解析 `Input Shapes` 生成 case selectors。产出 `kernel_worklist`，其中 `worklist.csv` 是给人看的（原 6 列 + `bucket` + `rank` + `excluded_reason`），`text.json` 是给下游任务读的。

同时把环境事实写进 `text.json` 的 `environment` 块：`gpu_target: gfx950`、`gpu_type: mi355x`、`framework: sglang`、镜像 tag 与 digest、ROCm 版本。这些值来自包变量，源头是 `glm53flash-demo/environment.md`。

### 8.3 `identify`

对每个选中算子调 `KernelSourceFinder.search(name)`。三个实现要点：

1. **显式传 repos，不用 auto_clone 的默认列表。** 默认列表不含 sglang（第 2.3 节）。`repos` 参数传入包变量指定的 sglang 检出路径、aiter 路径、tilelang 路径，加上 `amd_kernel_finder` 内建的几个。
2. **repos_base_dir 指向 `/data`，不指向 `/apps`。** `/apps` 只剩 3.7 T，而 clone pytorch + vllm + rocm-libraries 是几十 GB 量级。
3. **dtype 从符号名反推，作为 trace 缺失时的替代。** `mfma_moe1_silu_mul_afp4_wfp4_bf16_...` 这样的名字里 `afp4`（activation fp4）、`wfp4`（weight fp4）、`bf16`（输出）已经编码了三个 dtype。规则同样写在 `kernel_taxonomy.yaml` 里。反推不出的标 `dtype: unknown` 并进入 `missing_fields`。

在输入是 19 列增强 CSV 的情况下，跳过 1 与 2，直接读列。

### 8.4 `build_workset`

AI 任务，readme.md 是它的全部指令。给它的输入是 `kernel_worklist` 与 `operator_identity` 两个 staged handoff（通过 `$AGENT_SYS_INPUT_KERNEL_WORKLIST` 与 `$AGENT_SYS_INPUT_OPERATOR_IDENTITY` 读），输出写进 `$AGENT_SYS_OUTPUT_OPERATOR_WORKSET`。

readme.md 里必须包含的约束：

- `driver.py` 的 stdout 契约原文（第 4.3 节），一字不差地贴进去，因为这是硬格式。
- `graph_harness.py` 从 `KernelForge/examples/triton-softmax-forge-loop/` 原样复制，不要自己写计时逻辑。
- `naive_torch.py` 在 `baseline_ref_kind` 为 `eager_fn` 或 `perftest_wrapper` 时应当 import 现成实现而不是重写；只有 `inline_in_test` 或 `none` 时才自己写。
- 不许编造 `source_file`。`operator_identity` 里标了 `unknown` 的字段，在 `invocation_spec` 里保持空串并加进 `missing_fields`，在 `forge_task.yaml` 里写 `TODO` 注释。
- 每个算子至少 3 个 case，且必须覆盖 `operator_identity` 给出的全部 `case_selectors`（forge-loop preflight 会拒绝缺 case 的 driver）。

### 8.5 `verify_workset`

通过 `srun --jobid=<id> --overlap -N1 -n1 -w <node>` 进入 n05-21，在 `lmsysorg/sglang:v0.5.18-rocm720-mi35x` 容器里对每个算子执行：

```
python driver.py                                    → 解析 SNR / allclose
for i in 1..5:  python driver.py --warmup 10 --iters 20 --bench-mode   → 5 组
```

mission 3.2.7 要求"5 次加权平均，每次运行 loop 10 次以上取平均"，上面的循环形态直接对应。加权方式与 rsd 计算写在 `assets/lib/bench_stats.py`，`check_workset_runs` 复用同一个模块，避免生成与校验两处算法不一致。

容器需要 `--device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --security-opt seccomp=unconfined`，这组参数从 `glm53flash-demo/scripts/mix_worker.sh` 移植。

每个算子独立超时（包变量 `per_op_timeout_s`，默认 900）。一个算子失败不影响其余，失败信息写进 `workset_evidence` 的对应条目而不是让整个任务退出——`check_workset_runs` 再决定是否阻塞。原因是 program task 成功时 stdout 不被框架保留（`_detail()` 只在退出码非 0 时保留尾部 8 KB），所有需要留证的输出都要显式写进 handoff。

### 8.6 `packup`

按 `experiment-result-packup` 的 `deliverable_layout.md` 组装：README / REPRODUCE / environment / scripts / results / logs / notes。`collect_env.sh` 复用 skill 里那份，通过 srun 在 n05-21 上执行。

---

## 9. 包变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `jobid` | 无 | Slurm 作业 ID，每次都变，因此不给默认值 |
| `gpu_node` | 无 | 跑 `verify_workset` 的节点 hostname |
| `seed_csv` | `/apps/.../sglang_1p1d_glm5.2/profiles/20260826_053606/decode/megapie/gap_analysis/gap_analysis.csv` | mock 输入 |
| `magpie_root` | `/apps/.../Magpie` | `amd_kernel_finder` 的来源 |
| `sglang_src` | 无 | sglang 检出路径，供 `identify` 索引 |
| `aiter_src` | `/sgl-workspace/aiter`（容器内） | 同上 |
| `repos_base_dir` | `/data/agent_sys_analyze/repos` | `amd_kernel_finder` 的 clone 目标，必须在本地盘 |
| `work_root` | `/data/agent_sys_analyze` | 节点本地工作目录 |
| `image` | `lmsysorg/sglang:v0.5.18-rocm720-mi35x` | `verify_workset` 的容器 |
| `gpu_target` | `gfx950` | 传给 forge-loop |
| `gpu_type` | `mi355x` | 传给 forge-loop |
| `framework` | `sglang` | 传给 forge-loop |
| `top_n` | `5` | 选中算子数 |
| `min_pct` | `0.1` | 候选下限 |
| `min_calls` | `100` | 候选下限 |
| `max_cases` | `4` | 每算子 case 数上限 |
| `snr_threshold` | `30.0` | 正确性门槛，与 forge-loop 默认一致 |
| `min_kernel_rows` | `20` | `check_kernel_table` 阈值 |
| `max_unknown_ratio` | `0.3` | `check_worklist_shape` 阈值 |
| `min_resolve_ratio` | `0.6` | `check_identity_resolved` 阈值 |
| `max_rsd` | `0.1` | `check_workset_runs` 阈值 |
| `per_op_timeout_s` | `900` | 单算子验证超时 |

前两个不给默认值是刻意的：照 `single_real_task/README.md` 的做法，缺了会在加载期报出文件、行号和变量名，而不是在执行中途以别的形态失败。

---

## 10. 单任务调试 main

mission 要求每个步骤有"单独调试用的 single task main"。这里有一个框架限制需要说明：`agent_sys/cli/main.py:669` 与 `:688` 都是 `build.root_task("main", registry)`，根 closure 的名字**硬编码为 `main`**，CLI 没有指定入口的参数。而 `spec_loader/package.py` 会扫描包内任意位置的 YAML，同名对象会冲突。

因此单任务 main 只能是**独立的 package 目录**。方案是：

```
analyze-demo/                   完整包
analyze-demo-singles/
  ├── rank/
  │   ├── main.yaml             subgraph 只有 {closure: rank, froms: [], is_end: true}
  │   ├── steps/  -> symlink    指向 ../../analyze-demo/steps
  │   └── assets/ -> symlink    指向 ../../analyze-demo/assets
  ├── identify/
  ├── build_workset/
  └── verify_workset/
```

symlink 能否被 `spec_loader` 正确处理**尚未验证**，列为第 15 节的待确认问题。在 symlink 不可行的情况下，退路是每个 single 目录放一个 `main.yaml` 加一个生成脚本 `tools/sync_singles.sh` 做实体复制，复制产物加进 `.gitignore`。

对于只跑前几步的场景，还有一个更省事的做法：直接编辑 `analyze-demo/main.yaml` 的 subgraph，注释掉尚未实现的叶子。这与 `profiling-demo/DESIGN.md` 第 10 节的渐进式做法一致，适合开发期，不适合作为交付物。

---

## 11. 前置准备

不属于 package 本身，建议先手工完成并验证。

1. **安装 agent_sys**：`pip install -e agent_sys` 与 `pip install -e "agent_sys[claude]"`，然后 `agent-sys run --dry-run` 确认登录节点的 sandbox 存在性检查能通过。这台机器没有 bwrap，走 Landlock 路径，内核 6.8 应当满足，但未实测。这一项与 profiling-demo 共用。
2. **检出 sglang 源码**：`identify` 需要它来索引 GLM-5.3-Flash 的接入点。取 PR #36507 的 pinned head `7fa1924c04487feffc3f5f111a424ec1f0b22362`（来自 `glm53flash-demo/environment.md`），检出到 `/data` 而不是 `/apps`。
3. **预热 `amd_kernel_finder` 的仓库缓存**：首次运行会 clone pytorch / vllm / aiter / triton / rocm-libraries，几十 GB 且耗时长。建议先手工执行一次并确认 `repos_base_dir` 下的检出完整，之后 `identify` 就只是索引与查询。

`profiling-demo` 那三项前置（构建 glm53 镜像、准备 306 GB 权重、拉 AIPerf 镜像）对本包不适用，理由见第 3.3 节。

---

## 12. 已知风险与取舍

**`main_kernel` 解析不出设备符号对应的源文件，但这不再是风险。** 第 4.4 节的修正说明了原因：workset 的单位是框架层入口函数，不是设备 kernel 符号。Arena 的 TileLang 与 KDA 两个样例都是在 launcher 为 "Not found" 的前提下，直接把 `target_kernel_functions` 定位到框架层可调用入口。因此 `main_kernel` 的正确处理是"找到 sglang 里调用 TileLang DSA 的那个 Python 函数"，而不是"找到 `main_kernel` 的源文件"。

残留的真实风险是：这一步依赖 `build_workset` 的 AI 任务读 sglang 源码并做判断，质量不确定。约束手段是 `check_workset_shape` 要求 `target_kernel_functions` 非空、且能在 `source_file_path` 列出的文件里用 AST 定位到同名函数定义。定位不到即判负，不接受只写一个名字。

**`Avg time (us)` 不是 forge-loop 的 baseline。** CSV 里的平均耗时是服务运行中的统计，包含了排队、不同 batch size 混合、以及与其他 kernel 的资源竞争。forge-loop 的 `baseline_wall_ms` 需要的是独立 bench 的 wall time。两者可能差出数倍。因此 `forge_task.yaml` 的 `targets.baseline_wall_ms` 取自 `verify_workset` 的实测，而 CSV 里的数字放在 `provenance.json` 里作为"截取时 profile 报告中的性能"（mission 3.2.8），两者在 README 中明确区分。

**`deployment` 块的信息本阶段拿不到。** Hyperloom `invocation_spec` 的 `deployment.batch.serving_concurrency` 与 `deployment.sequence.request_tokens` 描述的是"这个算子是在什么服务负载下被观测到的"，数据源是 profiling 阶段的 AIPerf 报告，不在 `kernel_table` 里。两个选择：一是让 `profiling-demo` 的 `kernel_table` 额外携带一份负载摘要，二是把 `analyze-demo` 的 `rank` 的 `froms` 扩展到也包含 `aiperf_report`。倾向第二个，因为它不改变已有 kind 的定义。这一项列在第 15 节。

**AI 任务的产出质量不可预期。** `build_workset` 生成的 `driver.py` 可能不满足 stdout 契约，或者 `naive_torch.py` 与实际算子语义不符。`verify_workset` + `check_workset_runs` 是对它的实际约束：driver 跑不出 SNR 行、或 SNR 低于阈值，都会让 validator 判负。代价是失败要到流水线后段才暴露。作为缓解，`build_workset` 的 readme 里要求它自检生成的 driver 能被 `python -c "import ast; ast.parse(open('driver.py').read())"` 通过，并且 `--help` 里出现 `--bench-mode` 与 `--profile-run`。

**源文件快照的体量。** 第 5.3 节的 `kernel/snapshot/` 是可选项。aiter 的单个 asm kernel 源文件是几十 KB 量级，5 个算子不构成压力；但如果某个算子是多文件的（`--source-files` 场景），快照可能膨胀。倾向做法是只快照 `source_file` 本身加直接依赖，超过 `max_snapshot_kb`（默认 512）时退化为只记录路径与 commit。

**program task 没有超时。** `agent/backends/program.py` 只有轮询等待子进程，没有 wall-clock 上限。CLI 的 `_settle` 有 1800 秒上限，管的是"整张图是否静默"而不是单任务。因此 `identify`（clone 仓库）与 `verify_workset`（GPU 测试）的 entry.sh 各自带 `timeout`。

**Slurm 作业时限。** 28080 是 9 小时上限的 `hold-mi355x` 作业。本包一次完整执行预计 30–90 分钟（`identify` 首次 clone 是主要开销），单次执行得完，连续调试需要注意续期。

**SIKL 格式缺席。** mission 3.2.9 要求参考 SIKL 的算子定义格式，但本地没有该仓库，GitHub 上是否可访问未确认。本方案用 KernelForge task YAML + Hyperloom invocation_spec 两个已有格式替代。如果后续拿到 SIKL，增加一个第三份导出文件即可，`operator_identity` 里的字段是格式无关的。

---

## 13. 目录布局

```
analyze-demo/
├── DESIGN.md                        本文档
├── README.md                        跑法、图的形状、预期产物
├── main.yaml                        根 closure + subgraph
├── shared.yaml                      共享的 program agent + AI agent
├── steps/
│   ├── seed_table.yaml              kernel_table kind + check_kernel_table + seed_table task
│   ├── rank.yaml                    kernel_worklist kind + check_worklist_shape + rank task
│   ├── identify.yaml                operator_identity kind + check_identity_resolved + identify task
│   ├── build_workset.yaml          operator_workset kind + check_workset_shape + AI agent + task
│   ├── verify_workset.yaml          workset_evidence kind + check_workset_runs + task
│   └── packup.yaml                  analyze_packup kind + check_packup_shape + task
└── assets/
    ├── lib/
    │   ├── store.py                 从 demo 复制，validator 读 zone 内 JSON 用
    │   ├── remote.sh                srun 远程执行封装，与 profiling-demo 共用一份逻辑
    │   ├── csv_io.py                gap_analysis CSV 的读写与两种表头形态的判别
    │   ├── kernel_taxonomy.yaml     分类桶规则表 + dtype 反推规则
    │   ├── shapes.py                Input Shapes 解析与 case selector 生成
    │   └── bench_stats.py           5 组加权平均与 rsd，生成与校验共用
    ├── main.task/readme.md
    ├── seed_table.task/             readme.md + entry.sh + seed.py
    ├── rank.task/                   readme.md + entry.sh + rank.py
    ├── identify.task/               readme.md + entry.sh + identify.py
    ├── build_workset.task/         readme.md（AI 任务，无 entry.sh）
    ├── verify_workset.task/         readme.md + entry.sh + verify.py + run_in_container.sh
    ├── packup.task/                 readme.md + entry.sh + packup.py + collect_env.sh
    ├── templates/
    │   ├── graph_harness.py         从 KernelForge examples 原样复制，供 AI 任务引用
    │   ├── driver_contract.md       forge-loop stdout 契约原文
    │   └── forge_task.schema.yaml   KernelForge task definition 的字段说明
    ├── check_kernel_table.validator/
    ├── check_worklist_shape.validator/
    ├── check_identity_resolved.validator/
    ├── check_workset_shape.validator/
    ├── check_workset_runs.validator/
    └── check_packup_shape.validator/
```

**2026-09-01 的改名，本节以上文字保持原样。** 上面的 `packup` task、
`check_packup_shape` validator 以及对应的 `steps/packup.yaml`、
`assets/packup.task/`、`assets/check_packup_shape.validator/` 已分别改为
`pack_analyze`、`check_analyze_packup_shape`、`steps/pack_analyze.yaml`、
`assets/pack_analyze.task/`、`assets/check_analyze_packup_shape.validator/`。

原因是 `profiling-demo` 也有一个收尾的装配步骤，当初同样取名 `packup`，旁边同样有
`check_packup_shape`。`spec_loader/registry.py` 的注册表只以 `name` 为键（`version`
按 schema 的说明是维护元数据，不构成第二个槽位），所以两个包进入同一个 registry 时这两个
名字都会报 `SpecInconsistent`。

这两处不是同一概念的两种实现：两个 task 的 `inputs`、`outputs`、`agent` 与 body
脚本没有一项相同，两个 validator 分别读 `analyze_packup` 与 `profile_packup`，args 连
拼写都不同（`required_files` 对 `require_files`）。它们只是各自命名时都选了最自然的那个词，
所以改名就是全部修正，不是绕过。

`spec_loader/assets.py` 按 spec 名匹配 `assets/` 下的文件名，因此目录一并改名；两个
`entry.sh` 里写死的路径也随之更新。

`kernel_table` 与 `check_kernel_table` 没有改名：它们确实是同一个 kind 被声明了两次，
改名会留下两个 kind 而接图只需要一个。第 3.2 节与 `README.md` 的 "Joining up with
profiling-demo" 说明这一项如何处理。

### 运行方式

```
AGENT_SYS_NO_PERMISSIONS=1 agent-sys run \
  --package agent_sys/examples/llm_e2e_performance_optimization/analyze-demo \
  --var jobid=28080 \
  --var gpu_node=smci355-ccs-aus-n05-21 \
  --var sglang_src=/data/src/sglang
```

### 与上下游的衔接

上游：删掉 `seed_table` 叶子，把 `rank` 的 `froms` 改成 `[kernel_scan]`，`kernel_table` kind 的定义两包保持一致即可合图。

下游：`operator_workset` 里每个算子目录的 `run_forge.sh` 就是 kernel optimization 阶段的入口，它拼出的命令形如：

```
kernel-agents forge-loop \
  --workspace <worktree> --kernel <source_file> --driver <workset>/driver.py \
  --program-md-file <workset>/program.md \
  --invocation-spec-file <workset>/invocation_spec_<op>.json \
  --fellow <triton|ck|flydsl|hip>-fellow \
  --gpu-target gfx950 --gpu-type mi355x --framework sglang \
  --operator-name <logical_op> --snr-threshold 30.0 \
  --result-json <out>/forge_result.json
```

---

## 14. 落地顺序

分五步，每步都有可验收的中间态：

1. **对照 Arena 的三个 sglang 样例，手工做一个算子的完整 workset**（不涉及 agent_sys）。选样例 CSV 里 routable 第二名 `mfma_moe1_silu_mul_afp4_wfp4_bf16_...`（3.13%，66150 次调用，MoE GEMM），走完"符号名 → `amd_kernel_finder` 给方向 → 在 sglang/aiter 源码里定位入口函数 → 写 `config.yaml` / `session_cases.json` / 两个 driver → 在 n05-21 上实测正确性与性能"这条链路。**这是整个方案里不确定性最高的一环，建议最先做。** 它的产出有三个用途：验证第 4.4 节的"向上定位"是否可行、给 `build_workset` 的 readme 提供一个可引用的完整范例、以及标定 `check_workset_shape` 的各项阈值。
2. **只做 `seed_table` + `rank`**：两个纯本地任务，验证 handoff 写入、validator 判定、`--var` 传递这条链路。产出的 `kernel_worklist` 可以直接拿给人 review 分类规则是否合理。
3. **加 `identify`**：验证网络与仓库索引。这一步的产出直接决定 workset 的信息完整度。
4. **加 `build_workset`**：第一次接 claude code sdk 后端，验证 AI 任务读输入、写 handoff 这条路。
5. **加 `verify_workset` 与 `packup`**：接 GPU，补齐 README 与 REPRODUCE。

---

## 15. 需要确认的问题

第 1 节表格里的四项已确认，不再列入。剩余待确认：

1. **是否把 AgentKernelArena 的 `config.yaml` + `session_cases.json` 加为第三份产出**（第 5.1 节选项 C）。它是本地唯一携带容器仓库路径、可编辑文件白名单与分项超时的格式，也是 21 个真实样例背书的既有实践。增量成本低，因为数据来源与已确认的两份产出相同。
2. **SIKL**：已确认是 `AMD-AGI` 下的 private 仓库（组织可读、该 repo 返回 404），本机无副本。是否有人能提供访问权限？如果没有，第 5.1 节选项 C 是最接近 mission 3.2.9 要求的替代物；后续拿到 SIKL 时再增加一份导出即可，`operator_identity` 的字段是格式无关的。
3. **NCCL 的处理**：78.63% 的通信开销被排除在候选之外（第 6.1 节）。它会以 `bucket: collective, excluded_reason: not_routable_by_forge_loop` 留在 `kernel_worklist` 里。这个信息是否需要以别的形式向上游反馈，例如在 `kernel_worklist` 之外单独出一份并行策略观察？
4. **`deployment` 信息的来源**（第 12 节）：Hyperloom `invocation_spec` 的 `deployment.batch.serving_concurrency` 与 `deployment.sequence.request_tokens` 来自 AIPerf 报告，不在 `kernel_table` 里。倾向把 `rank` 的 `froms` 扩展到也包含 `aiperf_report`，这样不改动已有 kind 的定义。是否认可？
5. **单任务 main 的形态**（第 10 节）：`agent_sys/cli/main.py:669` 把根 closure 名字硬编码为 `main`，单任务调试只能是独立 package 目录。symlink 方案需要先验证 `spec_loader` 是否跟随符号链接。在不跟随的情况下，退路是"生成脚本 + 实体复制 + gitignore"，是否接受？
6. **`top_n = 5` 是否合适**：mission 没有规定数量。5 个算子对应 `verify_workset` 大约 15–75 分钟。
7. **是否直接复用 Arena 的三个 sglang 样例作为验收基准**：`mi355x_sglang_triton_mxfp8_grouped_gemm`、`mi355x_sglang_triton_mxfp8_linear`、`mi300x_sglang_hip_pa_decode` 是现成的完整 workset。让 `build_workset` 产出的结果与它们做结构对比，比自定义 validator 阈值更有说服力。
