# decode CUDA graph variant 的跨 DP rank 一致性：通用修法与验证（11-03 起）

接 `../car-copyin-141-20261103/`（根因定位）。用户 11-03 08:26 要求：用 138 / 140 做全面、准确的正确性验证；修法用适合提给 SGLang 的通用实现；dense 层 FP8 放后面。

## 问题（已在 141 上确认）

镜像（v0522，上游 main 同样如此）在 HIP 上为 DSA 模型的每个 batch size 捕获两套 decode graph：dense（`max_kv_len ≤ index_topk = 2048`）和 sparse
（`layers/attention/graph_variants.py` `DsaGraphVariants`）。**每个 DP rank 按自己 batch 的最长上下文各选一套**，没有请求的 rank 一律 sparse。
DP attention 下各 rank 逐层锁步，graph 里的集合通信跨 rank 配对；aiter 的 collective 在捕获时注册的是"本 graph 的输入地址"，
同一步里不同 rank 回放不同的 graph，就会读到对端另一套 graph 的旧缓冲。141 上：E11（graph 内走 aiter all-gather / reduce-scatter）乱码 26 / 200；
把所有 rank 固定到 sparse 后 0 / 200。

两个附带事实（读代码确认）：
1. variant 在 graph 里只影响 indexer 的 `_should_skip_logits_computation()`（dense 时跳过 logits，只写 K），而它只在 DECODE / IDLE 模式生效；
   **TARGET_VERIFY（开 MTP 时的主模型 graph）里两套 graph 执行的 kernel 完全相同**。开 MTP 的部署多捕获了一倍的 verify graph，没有任何收益。
2. `select()` 在 verify 路径上没有 host 侧长度，每次回放前做一次 `seq_lens.max().item()`，是 10-30 decode trace 里 GPU 空闲的最大单一来源（约 1 ms/步）。

## 修法（`patches/fix/`，6 个文件，对比 `patches/orig/`，基线 md5 见 `patches/orig/MD5SUMS`）

1. **只给 DECODE 模式的捕获建 dense / sparse 两套 graph**（`graph_variants.py` `create_attention_graph_variants` 增加 `capture_forward_mode` 参数，
   `decode_cuda_graph_runner.py` 传入）。TARGET_VERIFY 只捕获一套：开 MTP 时既没有跨 rank 错配，也没有每步的 `.item()`，verify graph 的捕获时间和显存减半。
2. **DECODE 模式下各 rank 用同一个长度选 variant**：DP attention 每轮已有的 MLP sync 表（`dp_attn.py` `MLPSyncBatchInfo`）加一列"本 rank decode 步的最长序列上界"，
   取各 rank 最大值，经 `ScheduleBatch` / `ForwardBatch` 的 `dp_decode_graph_max_seq_len` 传到 `select()`。照的是同表里已有的 `prefill_cuda_graph_max_prefix_len`
   （prefill graph 的跨 rank 协调）的写法。上界只读 host：有 `seq_lens_cpu` 时取精确值（与镜像的选择规则相同）；没有时用已提交长度加两步宽度。
   不加新的集合通信，也不读设备。没有 DP sync 的部署（单 rank）走原逻辑。

不做的：拷入（capture copy-in）也能修，但它在每次 collective 前多一次拷贝，而且只是绕开症状；固定 sparse 是定制化做法（用户 11-03 否决）。

提上游时要补的：上游 main 还有 `SeqLenBucketGraphVariants`、`Dsv41CandidateGraphVariants`，它们同样按本 rank 长度选，应一并读这个字段；
补 SGLang 风格的单元测试；把 v0522 上的改动对到 main。

## 验证计划（开跑前写下，11-03 09:20）

两台节点（prefill、decode，各用 GPU 4–7），跨机 RDMA，真实接受率，kit 为本目录 `kit/`（= `2p1d-kit/llr35` 精简版 + `KIT_NO_SPEC` 开关）。
在其他集群上的完整步骤见 **`REPRODUCE.zh-CN.md`**（11-03 12:30 起整理为自包含的包，不再依赖 crusoe 路径）。
每次部署都核对：decode 容器里 6 个文件的 md5（证明挂的是哪一版）、相关环境变量、`DSA dual-graph enabled` 日志行数与 graph 捕获耗时。
统计用 `scripts/probe_stats.py`（排除失败请求；开头坏不再把 "They" / "There" 和空输出算进去），所有被标记的样本写进 `*.review.txt` 逐条人工核对。

| 组 | 配置 | 内容 | 判据 |
|---|---|---|---|
| U | 单元测试 `scripts/test_fix.py`（镜像内，挂 fix） | 修法每一处的行为，`select()` 不读设备 | 全部通过 |
| A | `site-A-gate`：E10 限池门禁，镜像原样（对照） | 200 题 + 两遍 16k × 128，× 2 | 开头坏 + 乱码合计 ≥ 3（否则本次的限池门禁没有检验力，S1 / S2 判"不确定"） |
| B | `site-B-gate`：E10 限池门禁 + 修法 | 同上，× 3，与 A 交替 | 600 题开头坏 ≤ 1、乱码 ≤ 1；每次 `gate_pass2` PASS；每遍捞针 ≥ 62/64、长前缀 GSM8K ≥ 60/64；启动日志 0 行 dual-graph |
| C0 | `site-C0-e11`：E11，无修法（正对照） | smoke + 200 题 | 乱码 ≥ 10（11-02 同机对：44） |
| C | `site-C-gate`：E11 限池门禁 + 修法 | 同 B，× 2 | 400 题乱码 0、开头坏 ≤ 1；长上下文同 B |
| N0 | `site-N0-nospec`：不开 MTP、KF 关、E11 collective，无修法（第 2 部分正对照） | smoke + 200 题 | 乱码 > 0（为 0 则第 2 部分在此负载上"未观察到必要性"，如实记录） |
| N1a | N0 + 修法 + 审计（`patches/debug/`，每次回放读一次设备，只用于机制） | smoke + 400 题 + 16k × 32 | 新规则跨 rank 不一致 0、dense 时有超过 2048 的序列 0、上界被突破 0、无上界 0；dense 占比 > 0；同一负载下旧规则的跨 rank 不一致 > 0 |
| N1 | N0 + 修法 | smoke + 400 题 + 16k × 32 | 乱码 0、开头坏 ≤ 1 |
| B / C 生产 | `site-B-prod` / `site-C-prod`：不限池 | smoke + 1,319 题 + 16k × 128 + 100k × 32 + InferenceX lm-eval GSM8K | smoke 6/6；探测正确 ≥ 0.96、乱码 0、开头坏 ≤ 1；16k 捞针 ≥ 62/64、长前缀 ≥ 60/64；100k 各 ≥ 15/16；lm-eval strict / flexible ≥ 0.96 |
| 计分 | B / C 生产配置 C104，3600 s | 吞吐 | 预测 B ≥ 51,900（去掉每步同步）、C ≥ 52,400；同对 E10 51,374，单次波动约 ±1% |

检验力：开 CAR 的限池门禁此前 8 次里 6 次开头坏 3–8 / 200（约 2.5%）。若修法无效，B 的 600 题约有 15 条，≤ 1 条的概率约 5 × 10⁻⁶；
即使真实率只有 1%，≤ 1 条的概率也只有约 1.7%。

队列 `scripts/run_verify.sh`（顺序：preflight → unit → B 门禁 → N1a → A / B 交替 → C0 → C × 2 → N0 → N1 → B / C 生产 → 计分），
Slurm 下用 `scripts/slurm_verify.sbatch`；判定 `scripts/judge.py`（上表的判据，逐条打印 PASS / FAIL）。日志在 `$LOG_ROOT/`。

补充的两个核对（11-03 12:40）：
- **补丁基线**：本地参考副本 `~/.cache/llyq/sgl0522` 是基础镜像 `lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261018` 的 sglang，**不含** Infera Dockerfile
  打的补丁（例如镜像里的 `pool_host/mla.py` 多一段 `GLM52_ROCM_STAGED_WRITE_BACK`）。逐个读了 fdbafb60 的 `Dockerfile.sglang` 与它应用的补丁脚本：
  改动的是 `dsa_backend.py`、`dsa_indexer.py`、mooncake `conn.py`、`disaggregation/prefill.py`、`pool_host/{mla,dsa,mha}.py`、`forward_mla_rocm.py`、
  `kv_events_publisher.py`、`pd_disaggregation_hook.py`、`serving_responses.py`；`dp_attn.py` 只做标记检查不改。**修法的 6 个文件都不在其中**，
  所以它们在镜像里与参考副本相同（`graph_variants.py` 11-03 早上已在真实镜像上核对）。prefill 侧 5 个文件的镜像原版 md5 也写进了 `patches/BASE_MD5SUMS`
  （`prefill.py` 由 v3 补丁反推得到），`run_verify.sh` 的 preflight 在两台上逐个核对。
- **配方等价**：`config/recipe-e10.env`、`recipe-gate.env` 与 crusoe 上的原配置（`../prefill-opt-20261027/config/site-llr35-v3-evict512k-bq16.env`、
  `site-llr35-v3-evict-bq16-gate.env`）在同一组站点值下展开后，所有部署变量逐项相同（kit 路径除外）。

## 结果

**镜像重建与单元测试（11-03 12:51–13:01，crusoe Slurm 作业 261868，`crsuse2-m2m-v3-073`，`scripts/slurm_image.sbatch`）**：按 fdbafb60 重建出的镜像
ID 是 `7b38182c…`（与 138 上的 `509b8427` 不同，层不同在意料之中），但 **11 个基线文件的 md5 全部与 `patches/BASE_MD5SUMS` 一致**（包括由 v3 补丁反推的
`prefill.py`），说明按 REPRODUCE 第 3.2 步重建即可原样使用这些补丁；`scripts/test_fix.py` 在该镜像里 **26/26 通过**。镜像存为
`/shared_nfs/liyingli/images/infera-sglang-v0522-fdbafb60.tar`（23 GB）。该节点空闲时 8 张卡显存均为 0、无 KFD 进程（12:38 的 v3-039 显示 10–11%，是那台节点的情况）。

部署验证（A / B / C / N 组与计分）尚未在任何集群上跑；用户 11-03 13:03 决定去另一个集群验证。

（以下为 12:40 的记录：11-03 08:38 起旧节点名 `crsuse2-m2m-138 / 140 / 141` 从 DNS 消失：集群改为 Slurm（spur）管理，节点改名 `crsuse2-m2m-v3-*`，
v3-138 / v3-140 已分给别人的作业。12:38 用 `amd-agentx` + `amd-burst-qos` 的 1 节点探测作业 261858 拿到 v3-039：docker 可用、RDMA ionic_0–7、
模型路径都在；但没有 `infera-sglang:v0522-fdbafb60` 镜像（要按 fdbafb60 重建），空闲 GPU 显存显示 10–11%（原因待查）。）
