# CRS PP4 基线复测，2026-10-09

按 [task.md](task.md) 在 CRS 节点复测。本次结果只能来自新的完整运行；历史被抢占运行的实时统计不作为正式结果。

## 测试口径

| 项目 | 参数 |
|---|---|
| Prefill | 136，GPU 0–3 / 4–7，2 个 TP1×PP4，层划分 20/20/20/18 |
| Decode / 控制节点 | 138，GPU 0–3，TP4 + DCP4，EP1 |
| 模型 | `/shared_nfs/models/GLM-5.2-MXFP4` |
| 基础镜像 | `rocm/atom-dev:nightly_202609221542`，用户已确认沿用固定版本 |
| Dockerfile / 补丁 | `../../8p4d-atom-infera/docker/Dockerfile` 与同目录下 `patch/` |
| 新镜像标签 | `infera-atom:pp4-baseline-202609221542-20261009` |
| 引擎参数 | 保持 `config.sh` 的 PP4 参数、LMCache 160 GiB、block 16、显存比例 0.85/0.94 |
| InferenceX | `918524ff94045b3f091115f1051c22a8588edf2b` |
| AIPerf 子模块 | `754356e9a39acc6cc6afb242d123bb57c3fb6f75` |
| 正式测试 | C48/C80/C120，各自冷启动，每 lane 预热 10，测量 3600 s，MTP K3 / forced acceptance 2.99 |
| 正确性验证 | `MTP_AL=`，completions、流式 chat，GSM8K 5-shot 200 题 |

136 IP 为 `10.245.154.168`，138 IP 为 `10.245.157.237`。两节点均为 MI355X、amdgpu 6.14.14、ionic 25.08.4.004。
RDMA GID 1 为 IPv6，同编号 ionic 设备对应同 rail；端到端 ATOM RDMA 已在本次正确性验证和正式测量中使用。

## 运行与归档

两节点 `/shared_nfs` 只读，源码和模型从 NFS 读取，运行目录在 138 本地：
`/mnt/m2m_nobackup/xiaobche/yaoc-pp4-baseline-20261009`。
`TMP_DIR` 指向该目录，`CACHE_DIR=$TMP_DIR/cache`，`RESULTS_DIR=$TMP_DIR/results`。
本次覆盖参数保存在 `.tmp/crs.env`，在 138 以 `xiaobche` 执行：

```bash
cd /shared_nfs/yaoc/work/infera-test/Infera-atom-test/yaocheng/8p4d-atom-infera-prefill-pp4
source .tmp/crs.env
bash scripts/build_image.sh
bash .tmp/prepare_agentx.sh              # 仅 CPU，准备全部三档
PREPARE_ONLY=1 bash .tmp/gsm8k.sh         # 仅加载数据集和 tokenizer
# GPU 资源交接完成后执行：
bash scripts/up.sh CONC=48 MTP_AL= RUN_ID=crs-acc-c48
bash scripts/smoke.sh
bash .tmp/gsm8k.sh
# 确认 GSM8K 达标后：
bash scripts/down.sh
nohup setsid bash scripts/sweep.sh SWEEP_ID=crs-pp4 > "$TMP_DIR/logs/sweep.log" 2>&1 < /dev/null &
```

从有 NFS 写权限的登录节点取回日志、运行原始数据、镜像清单到本套件 `.tmp/`，正式聚合结果放入 `results/`。
构建时记录 base digest、最终 image ID、ATOM commit、包版本；两个节点分别构建，需比较包清单和补丁，不能仅凭相同标签认定内容相同。

本次 `.tmp/crs.env` 的必要覆盖参数如下，其余引擎参数由 `config.sh` 提供：

```bash
export PREFILL_NODE=crsuse2-m2m-136 PREFILL_IP=10.245.154.168
export DECODE_NODE=crsuse2-m2m-138 DECODE_IP=10.245.157.237
export CONTROL_NODE=crsuse2-m2m-138 CONTROL_IP=10.245.157.237
export MODEL=/shared_nfs/models/GLM-5.2-MXFP4
export TMP_DIR=/mnt/m2m_nobackup/xiaobche/yaoc-pp4-baseline-20261009
export CACHE_DIR=$TMP_DIR/cache RESULTS_DIR=$TMP_DIR/results
export IMAGE_BASE=rocm/atom-dev:nightly_202609221542
export IMAGE=infera-atom:pp4-baseline-202609221542-20261009
export PREFIX=yaoc-glm52-8p4d-pp4
export POINTS="48 80 120" DURATION=3600 WARMUP_PER_LANE=10
```

## 验收

- 镜像导入 `atom`、`infera`、`mooncake.engine` 成功；依赖检查如有问题记录具体原因。
- 路由有 2 个 prefill 和 1 个 decode；端到端 smoke 与 RDMA 日志正常。
- GSM8K 准确率至少 0.931（参考值 0.961，允许相差 3 个百分点）；forced acceptance 关闭。
- 三档各有 `agentx_conc<N>.json`，测量时长至少 3528 s，失败比例低于 10%，GPU 元数据为 8P+4D。
- 保存吞吐、TTFT、ITL、请求数、失败数和运行时长至汇总表；未通过或不完整的运行明确列为失败，不充当正式基线。

## 当前状态

不使用 GPU 的准备已完成：

- 两节点镜像构建和导入检查通过，包清单相同；ATOM 为 `d9f0720e2f99`，base digest 为 `8d7ebab3069ad…`。
- `pip check` 的 10 个问题全部存在于原始基础镜像，本次构建没有新增问题，详情见 `.record/issues.md` 第 9 条。
- C48/C80/C120 的 `PREPARE_ONLY=1` 均成功，`runtime.env` 与 `replay-command.txt` 已取回 `.tmp/prepared/c<N>/agentx/`；
  核对了 3600 s、预热 10、8P+4D、forced acceptance 2.99 与 4 个指标端点。
- AgentX 数据集缓存约 1.8 GiB，snapshot 为 `23f152f6f0f9399a85901b89a6458def0ef16729`；
  GSM8K 加载 7473 条训练数据、1319 条测试数据，模型 tokenizer 和 chat template 可正常加载。
- 构建与准备日志已同步到 `.tmp/logs/`；完整版本和补丁摘要见 `.record/crs-image-manifest.json`。

以上为无 GPU 准备阶段的记录。用户随后授权 GPU 测试并确认 C48/C80/C120 各 3600 s；
12:58 UTC 起在空闲的 136/138 执行。正确性验证通过：基础请求、流式、6 万 token needle，
以及关闭 forced acceptance 的 GSM8K 5-shot 200 题（195/200，97.5%）。
C48/C80/C120 正式测量均已完成，三档数据有效性检查通过；原始数据已归档并校验，测试容器已清理。结果与验收详见
[`../results/crs-20261009-report.md`](../results/crs-20261009-report.md)。
