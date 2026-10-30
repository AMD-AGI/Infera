# ROCm 10.2 SGLang 镜像验证报告：Infera build → 起服务 → 冒烟测试

> 实验记录，随进展实时更新。最新状态见「进度总览」，细节见「实验日志」（按时间倒序）。

## 结论（2026-10-30）

**ROCm 10.2 SGLang 镜像上，Infera 的 build → 起服务 → 冒烟测试全流程已跑通，mixed 和 PD 两个场景的 smoke 都是 6/6 通过。**

- 镜像：`rocm/ufb-private:sglang-v0.5.21-ubuntu24.04-py3.14-prereleases-device-all-rocm10.2.0rc0-aaaf7c860d`（官方未发布 rocm10.2 SGLang 镜像）
- 构建：现有 `Dockerfile.sglang` 不能直接换 base；新增 `Dockerfile.sglang.rocm102` 及 2 个 v0.5.21 适配补丁、修 1 处 Mooncake 构建脚本（第 4 节）
- mixed（TP4 + DP4-attention + MTP + HiCache，GPU 0-3）：smoke 6/6（第 5 节）
- PD（1P TP1×PP4 + TP4/DP4 decode，同机，Mooncake RDMA 传 KV）：smoke 6/6，另加 2 条全新 ~99.5k token needle 也正确（第 6 节）
- 需要关注：① 首次冷启动 / 首批长请求有一次性 AITER + Triton JIT，耗时明显（mixed 首次 healthy 约 24 分钟；PD 首个 66k 请求 530 s，之后同量级全新请求 4.4 s）；② rocm10.2 base 为 rc0 预发布 + 内部私有仓库；③ 本次只做冒烟，未跑 GSM8K / AgentX 性能


- 代码：`AMD-AGI/Infera`，分支 `verify/rocm10.2-sglang`（基于 main `a470fcb7`）
- 本地目录：`/home/liyingli/bench_agentx/baseline/Infera-rocm102`（NFS 共享，计算节点同路径可见）
- 测试节点：`crsuse2-m2m-141`（8×MI355X，gfx950），所有拉镜像 / build / 起服务均在此节点执行
- 模型：`/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`
- 节点产物目录：`141:~/rocm102-verify/`

## 进度总览

| # | 步骤 | 状态 | 备注 |
|---|---|---|---|
| 1 | 确认 rocm10.2 镜像来源 | ✅ 完成 | 官方未发布；用 `rocm/ufb-private` |
| 2 | 选择测试节点 | ✅ 完成 | crsuse2-m2m-141 |
| 3 | 拉取 base 镜像 | ✅ 完成 | 48 GB（解压后） |
| 4 | base 镜像布局检查 | ✅ 完成 | 见下文「base 镜像差异」 |
| 5 | Infera image build | ✅ 完成 | `infera-sglang:rocm102`（id `424e7c622483`，48.6 GB），见第 4 节 |
| 6 | mixed（`examples/sglang_mix_glm5.2`）起服务 + smoke | ✅ **smoke 6/6 通过** | GPU 0-3，见第 5 节 |
| 7 | PD（`examples/sglang_2p1d_glm5.2`，1P + 4 卡 decode 单机）起服务 + smoke | ✅ **smoke 6/6 通过** | prefill GPU 0-3，decode GPU 4-7，见第 6 节 |
| 8 | 清理已合入 PR 的目录 | ⚠️ 需手动 | `rm -rf` 被权限拒绝，见末尾 |

## 1. 镜像选择

**官方公开仓库没有 rocm10.2 的 SGLang 镜像**（截至 2026-10-30）：`lmsysorg/sglang-rocm`、`rocm/sgl-dev` 最新为
ROCm 10.0（`v0.5.22-rocm10-mi35x-20261019`）和 ROCm 10.1（`rocm101`，仅 mi45x）。

`rocm/ufb-private` 中 sglang 相关 rocm10.x tag 统计：rocm10.0 ×14、rocm10.1 ×107、**rocm10.2 ×99**、rocm10.3 ×19。
rocm10.2 中 sglang 版本最高：

| tag | sglang | 说明 |
|---|---|---|
| `sglang-mi450-v0.5.22-rocm10.2.0a20261013-bkc.20261014-baa9befa38` | v0.5.22 | **仅 mi450**，MI355X 不适用 |
| `sglang-v0.5.21-…-nightlies-device-all-rocm10.2.0a20261022-088f7d8eed` | v0.5.21 | 10-22 构建；tag 写 py3.14 实际 venv 为 python3.12 |
| `sglang-v0.5.21-…-prereleases-device-all-rocm10.2.0rc0-088f7d8eed` | v0.5.21 | 10-23 构建；同上 python3.12 |
| **`sglang-v0.5.21-ubuntu24.04-py3.14-prereleases-device-all-rocm10.2.0rc0-aaaf7c860d`** | v0.5.21 | **选用**：10-29 构建（最新），rocm10.2.0rc0，python3.14 |

选用镜像：`rocm/ufb-private:sglang-v0.5.21-ubuntu24.04-py3.14-prereleases-device-all-rocm10.2.0rc0-aaaf7c860d`
（image id `c15d804f274f`）。拉取方式：在 141 上 `docker login -u rocm --password-stdin` 后 `docker pull`。

## 2. 测试节点选择（2026-10-30 探测）

| 节点 | ssh | docker | GPU | 结论 |
|---|---|---|---|---|
| 135 | ✓ | ✓ | GPU4-7 满载 | 占用 |
| 136 | ✓ | ✓ | 8 卡显存占满 | 占用 |
| 137 | ✓ | ✓ | 空闲 | 有他人容器在跑，不用 |
| 138 | ✓ | ✓ | GPU4-7 占用 | 占用 |
| 139 | ✓ | ✗ 无 docker daemon | GPU0 占用 | 不可用 |
| 140 | ✗ publickey 拒绝 | — | — | 不可用 |
| **141** | ✓ | ✓ | **8 卡全空闲** | **选用**；docker root 26T 空闲，内存 2.7 TB，所需端口空闲，ionic_0..7 RDMA 在位 |
| 142 | ✓ | ✓ | GPU0-1 占用 | 部分占用 |

## 3. base 镜像差异（相对 `Dockerfile.sglang` 当前 pin 的 `lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261018`）

| 项 | 现有 pin（v0.5.22 / rocm7.2） | rocm10.2 镜像 |
|---|---|---|
| sglang | 0.5.22.dev20261018+g8dee3f6bdb | **0.5.21+amd.march.gaaaf7c860d**（`/sgl-workspace/sglang`） |
| ROCm | 7.2.0，`/opt/rocm` | **10.2.0rc0**，pip SDK `_rocm_sdk_devel`，`/opt/rocm` 为指向它的软链 |
| Python | 3.12 系统 | **3.14.10**，`/opt/venv` |
| torch | rocm7.2 | 2.14.0+rocm10.2.0rc0 |
| triton | — | 3.8.0+rocm10.2.0rc0 |
| aiter | — | amd-aiter 0.1.22.dev120（`/sgl-workspace/aiter`，JIT 目录 `/sgl-workspace/aiter/aiter/jit` 存在） |
| transformers | — | 5.12.1 |
| Mooncake | base 自带 | base 自带（`/opt/venv/lib/python3.14/site-packages/mooncake`） |
| 工具链 | — | cargo、cc、patch、git 均有；libclang 在 `_rocm_sdk_devel/lib/llvm/lib` |
| libionic | — | 未装（libibverbs1 50.0） |

预期风险：`Dockerfile.sglang` 的 v0522 补丁以 `--fuzz=0` / 精确锚点应用，针对 v0.5.22；在 v0.5.21 base 上可能有锚点不匹配，需要逐个判断（跳过 / 换用上一代补丁 / 重切）。

## 4. Infera image build 适配

在 141 上构建，`DOCKER_BUILDKIT=0`（见问题 0）。日志：`141:~/rocm102-verify/build-try*.log`。
补丁逐步 dry-run 脚本 `verify_rocm10.2/patch_dryrun.sh`，结果：`patch_dryrun.base.log`（原 v0522 补丁集）、`patch_dryrun.rocm102.log`（适配后）。

### 遇到的问题与处理

| # | 问题 | 原因 | 处理 |
|---|---|---|---|
| 0 | `docker build` 报 `unknown flag --progress` / `unknown shorthand flag -f` | 141 的 Docker 29.6.1 CLI 中 `build` 是 buildx 别名，但 CLI 未识别 `/usr/libexec/docker/cli-plugins/docker-buildx`（插件本身可执行，v0.35.0） | 用 legacy builder：`DOCKER_BUILDKIT=0 docker build`（Dockerfile 无 BuildKit 专有语法）。节点环境问题，未改仓库 |
| 1 | Mooncake 构建：`ninja: unknown target engine.cpython-310-…so` | `build_mooncake_sglang.sh` 硬编码 cp310；新 base 为 Python 3.14 | 改为由 `sysconfig EXT_SUFFIX` 推导（对旧 base 行为不变） |
| 2 | DSA 补丁校验失败：`glm4_moe.py :: NextN … fused_shared_experts_architecture -> False` | v0522 脚本把该修复当作「v0.5.22 已 upstream」只做断言；v0.5.21 没有 | 新增 `sglang_dsa/v0521/patch_glm_nextn_fused_shared_experts_arch.py`，在 DSA 脚本前补上这一行类属性（与 v0.5.22 upstream 相同） |
| 3 | `patch_prefill_nonfinal_chunk_wait_event.py`：upstream mooncake wait_event consumer missing | v0.5.21 的 prefill.py 已记录 early-send event，但 mooncake/conn.py **不等待它**（v0.5.22 才有 sglang#41395）；旧版完整补丁的 prefill.py 锚点因 assert 被重排成多行而失配 | 新增 `sglang_disagg/v0521/patch_mooncake_wait_event_consumer.py`：只跑旧完整补丁的 utils.py + conn.py 两部分，再跑 v0522 的 prefill.py 脚本。**这是 PD chunked prefill 正确性补丁，不能跳过** |
| 4 | `patch_mla_absorb_bmm_default_config.py`：anchor not found | 该 AMD v0.5.21 构建里没有 `_absorb_bmm_config` 函数，补丁要规避的崩溃路径不存在 | rocm102 Dockerfile 中去掉此补丁 |
| 5 | `patch_decode_radix_spec.py`：unsupported guard | v0.5.21 的 guard 改为 `not in (None, "DSPARK")` | rocm102 Dockerfile 默认 `APPLY_SGLANG_DECODE_RADIX_SPEC_PATCH=0`。infera 只有在 `SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC=1` 时才会在投机解码下开 decode radix，示例都不开，无影响 |

### build 结果（try3）

- 构建命令（141 上，仓库根目录）：`DOCKER_BUILDKIT=0 docker build -f deploy/docker/Dockerfile.sglang.rocm102 -t infera-sglang:rocm102 .`
- 产物 `infera-sglang:rocm102`，image id `424e7c622483`，48.6 GB
- 构建日志中全部补丁打印 patched/verified；DSA 字节码校验通过；Mooncake 重编通过 dma-buf / cross-host routing / RDMA 特性检查
- 镜像自检：`import infera, sglang, mooncake.engine` OK；`infera-router --help` OK；libionic1 = 54.0-187-1（ABI 4）
- 提交：`e674aaae`（Dockerfile + 补丁 + mooncake 脚本）、`fd164347`（2p1d worker.sh 同机 decode pid 偏移），均带 Signed-off-by，**未 push**

其余补丁（DSA indexer rows、dsa_dp_sync、dsa_page_table_rows v0522、responses tool args、hicache staged write-back、kv-metrics noblock、PP embed dim）在 v0.5.21 上原样可用。

### 仓库改动（分支 `verify/rocm10.2-sglang`）

- `deploy/docker/Dockerfile.sglang.rocm102`（新）：基于 `Dockerfile.sglang`，只改上表 2-5 涉及的步骤与 base；不动原 pinned 的 `Dockerfile.sglang`
- `deploy/docker/scripts/build_mooncake_sglang.sh`：Python 扩展名后缀自动推导
- `deploy/docker/patches/sglang_dsa/v0521/…`、`deploy/docker/patches/sglang_disagg/v0521/…`（新）
- `examples/sglang_2p1d_glm5.2/engine/worker.sh`：decode 与 prefill 同机时也先消耗 30,000 个 pid（避免 RCCL 共享内存名冲突，同 README note 4）

## 5. mixed 场景结果（`examples/sglang_mix_glm5.2`，TP4 + DP4-attention + EAGLE MTP + HiCache 270 GB/rank）

- 配置：`verify_rocm10.2/site.mix.sh`；kit 配方未改；真实 MTP 接受率（`SIM_ACCEPT_LEN` 为空）
- 运行目录：`141:~/rocm102-verify/work/mix/runs/20261030T044721Z/`（`smoke/`、`logs/` 均在此，down.sh 已保存全部容器日志）
- 起服务：etcd → engine → router 全部健康；router 注册 1 个 `mixed` worker，`dp_size=4`
- engine 启动耗时（首次冷启动，含 AITER JIT）：load_weight 605 s，scheduler_e2e 1217 s，CUDA graph target_verify 305 s / draft_decode 66 s；
  HTTP 起来后首个健康检查请求又触发了一轮 AITER JIT，总计约 24 分钟才 healthy（README 写 10-15 分钟，差在首次 JIT；JIT 结果缓存在 named volume，再启动会快）
- engine 日志无 Traceback。`[ERROR] … not documented` 为 transformers 5.12 的 docstring 告警，无害

| smoke 检查 | 结果 | 耗时 | 输出 |
|---|---|---|---|
| workers | ✅ | 0.01 s | mixed/dp4 |
| completion | ✅ | 1.24 s | " Paris. …" |
| chat | ✅ | 0.55 s | 391 |
| stream | ✅ | 0.57 s | 899 |
| long_prefix_cold（~66k token needle） | ✅ | 24.4 s | CEDAR-4729 |
| long_prefix_repeat（前缀缓存命中） | ✅ | 1.02 s | CEDAR-4729 |

**6/6 通过。** 备注：cold 66k 用时 24.4 s，kit README 中 rocm7.2/v0.5.22 镜像为 9.8 s；本次是冷启动后第一批请求，可能仍含 JIT，冒烟阶段不据此下性能结论。

## 6. PD 场景结果（`examples/sglang_2p1d_glm5.2`，1P + 4 卡 decode，单机）

- 形态：kit README 中的「1P with a 4-GPU decode (8 GPUs)」，但 prefill 与 decode 同在 141（无第二台整机空闲）：
  prefill0 GPU 0-3（TP1 × PP4，HiCache 250 GB/stage，EAGLE），decode GPU 4-7（TP4 + DP4-attention，EAGLE MTP 5/1/6），
  KV 走 Mooncake RDMA（ionic_0..7，GID idx 1），router kv-aware
- 配置：`verify_rocm10.2/site.pd.sh`；kit 配方未改，只有 worker.sh 同机 decode 的 pid 偏移（提交 `fd164347`）
- 运行目录：`141:~/rocm102-verify/work/pd/runs/20261030T052444Z/`（`smoke/`、`logs/`）
- 起服务：约 10 分钟 healthy（AITER JIT 已有一部分缓存）。router 注册 1 prefill + 1 decode
- 启动耗时：prefill load_weight 18.7 s / scheduler_e2e 187 s；decode load_weight 38.8 s / scheduler_e2e 492 s（CUDA graph target_verify 340 s）
- 同机两引擎未出现 RCCL `ncclCommInitRank` 冲突（pid 偏移生效）

| smoke 检查 | 结果 | 耗时 | 输出 |
|---|---|---|---|
| workers | ✅ | 0.01 s | decode prefill |
| completion | ✅ | 2.34 s | " Paris. …" |
| chat | ✅ | 5.61 s | 391 |
| stream | ✅ | 0.39 s | 899 |
| long_prefix_cold（~66k token needle） | ✅ | **530.6 s** | CEDAR-4729 |
| long_prefix_repeat | ✅ | 6.23 s | CEDAR-4729 |

**6/6 通过**；长 prompt 被切成多个 8k chunk 经 PP4 + Mooncake 传给 decode 后 needle 仍取对，说明 chunked prefill 的 KV 传输正确（这正是问题 3 的 wait-event 补丁所保护的路径）。

**long_prefix_cold 530 s 的排查：**
- 该请求期间 prefill / decode 都在按需 JIT 编译 AITER 模块（`mha_varlen_fwd_bf16…` 25 s、`module_rope_2c_cached_positions_fwd` 47 s、`module_moe_asm`、`module_cache`、`module_quant` 等），这是镜像首次遇到长 prompt 形状
- 之后用 `verify_rocm10.2/fresh_long_prompt.py` 发两条**全新**（不可能命中前缀缓存）的 ~99.5k token needle prompt：

| 请求 | prompt tokens | 耗时 | 结果 |
|---|---|---|---|
| fresh1 | 99,485 | 108.6 s | ✅ BIRCH-4215 |
| fresh2 | 99,505 | **4.44 s** | ✅ ALDER-5382 |

  fresh1 期间无 AITER 编译日志，但 prefill 时间线上第 1→2 个 chunk 间停顿 25 s、PP3 停顿 8 s，之后每个 chunk < 1 s，判断为 Triton 首次遇到新形状的编译；fresh2 稳态 4.4 s。
  **结论：530 s / 108 s 是一次性预热成本，稳态正常。** 生产部署建议上线前用长 prompt 预热，或持久化 JIT 缓存 volume（kit 已按 image id 建 named volume）。

**日志中的非致命 Traceback（prefill，共 5 个，2 类）：**
1. `/freeze_gc` Connection refused（05:28:48）：SGLang 预热后 best-effort 的 GC freeze 请求早于 prefill HTTP 端口监听，仅告警，服务正常
2. infera readiness 端口 `30090` address already in use：infera 的 best-effort readiness 探针端口默认 30090，同机 prefill 与 decode 抢同一端口，decode 先占到。router 与 smoke 不依赖该端口；kit 原本的 2 prefill 同机布局也会遇到。如需要可给其中一个容器设 `INFERA_READINESS_PORT`。属于单机共置的副作用，非 rocm10.2 相关
- decode 日志无 Traceback

## 实验日志（倒序）

### 2026-10-30 ~05:50 — PD smoke 6/6 通过；排查 cold 530 s；down.sh 完成
- 详见第 6 节。全部 r102 容器已删除，141 上无残留容器（显存由驱动延迟释放）。


### 2026-10-30 ~05:25 — 启动 PD
- GPU 显存约 6.5 分钟后全部释放。ionic GID idx1 为可路由 RoCEv2（与 kit 默认 `MC_GID_INDEX=1` 一致），8 个口均 ACTIVE。
- `site.pd.sh`：PREFILL_WORKERS=1，prefill0 GPU 0-3（TP1×PP4，HiCache 250 GB/stage），decode GPU 4-7（TP4 + DP4-attention），DECODE_SSH 指向本机，CTR `r102_pd`。
- 启动日志：`141:~/rocm102-verify/pd-up.log`。


### 2026-10-30 05:15 — mixed smoke 6/6 通过，down.sh 完成
- down 后 GPU1-3 显存仍未释放（驱动延迟释放，kit README note 3），等释放后再起 PD。


### 2026-10-30 ~05:30 — build try3 成功；启动 mixed
- site 配置 `verify_rocm10.2/site.mix.sh`：NODE_IP 10.245.148.188、GPU 0-3、CTR `r102_mix`、WORK_DIR `~/rocm102-verify/work/mix`
  （141 的 `/mnt/m2m_nobackup` 仅 root 可写，WORK_DIR 改放 NFS home；smoke 产物很小）。
- 启动日志：`141:~/rocm102-verify/mix-up.log`。

### 2026-10-30 ~05:10 — build try3 开始（`Dockerfile.sglang.rocm102`）
- 补丁 dry-run 在 base 容器内全部 OK（8/8 步，DSA 字节码校验通过）。

### 2026-10-30 ~04:50 — build try2 失败于 DSA 步（问题 2），dry-run 定位到问题 2-5
- try2 已通过 Mooncake 重编（问题 1 修复生效）、libionic、`pip install .[sglang]`。

### 2026-10-30 ~04:40 — build try1 失败于 Mooncake（问题 1）
- 之前约 20 分钟浪费在问题 0：后台 build 命令实际未启动。

### 2026-10-30 04:20 — base 镜像检查完成，开始 build
- 见第 3 节。

### 2026-10-30 04:19 — 镜像拉取完成
- 48 GB，image id `c15d804f274f`。

### 2026-10-30 ~03:58 — 准备工作
- clone Infera → `Infera-rocm102`，建分支 `verify/rocm10.2-sglang`。
- 选定 141，确认模型路径、端口、RDMA。
- PD 方案：只有 141 整机空闲，采用示例 README 已列出的「1P + 4 卡 decode（8 GPU）」形态，单机部署
  （`PREFILL_WORKERS=1`，`PREFILL0_GPUS=0,1,2,3`，`DECODE_GPUS=4,5,6,7`，`DECODE_NODE_IP=PREFILL_NODE_IP`）。
  同机两引擎 RCCL 共享内存按 pid 命名会冲突（2P1D README note 4），需给同机 decode 也加 pid 偏移（与 prefill 1 现有做法一致）。

## 目录清理（PR 状态）

| 目录 | 分支 / PR | 状态 | 处理 |
|---|---|---|---|
| Infera-mix-v0522 | #210 | MERGED，本地 HEAD = PR head，无本地改动 | **可删，需手动**：`rm -rf Infera-mix-v0522` |
| router-pr213 | #213 | MERGED，本地 HEAD = PR head，无本地改动 | **可删，需手动**：`rm -rf router-pr213` |
| Infera-kvcfg | AMD-AGI/Infera#222 | OPEN | 保留 |
| inferencex-app | SemiAnalysisAI/InferenceX-app#1542 | OPEN | 保留 |
| inferencex-infera | SemiAnalysisAI/InferenceX#4629 | OPEN | 保留 |
| sglang-cookbook | sgl-project/sglang#47117 | OPEN | 保留 |
| Infera-e2e-ssh | 无 PR，1 个提交不在 main | — | 保留，待确认 |
| srt-slurm-infera | 未找到 PR | — | 保留，待确认 |
| srt-slurm-ae3 / Infera | 有未提交改动 | — | 保留 |
