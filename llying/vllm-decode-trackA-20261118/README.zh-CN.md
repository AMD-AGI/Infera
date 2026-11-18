# A 线：GLM-5.2 decode 侧 kernel / 通信（vLLM v0.34.0 ROCm，聚合模式，v3-240）

接手 `../vllm-kf-port-20261117/HANDOFF-trackA.zh-CN.md`。节点 v3-240（作业 295840，8×MI355X），工作目录
`/home/liyingli/bench_agentx/vllm-trackA-20261118/`（原始日志、详细 bench JSON、完整 trace 在那里，本目录只放脚本和派生结果）。

## 0. 结论摘要

| 项 | 结论 | 证据 |
|---|---|---|
| A1 #47897 为什么前两次没生效 | 第二次的推测成立：覆盖层在 forward 时调 `get_current_vllm_config()`，worker 里此时没有配置上下文，抛 `AssertionError`，被 `except Exception` 吞掉后组大小为 0，且被 `functools.cache` 永久缓存 → 永远回落 aiter | `config/vllm.py::get_current_vllm_config`（未设置时 raise） |
| A1 引擎内生效 | 设 `VLLM_ROCM_DSA_GLUON_LPT_GROUP_ROWS=4` 后，日志出现 `Using the gfx950 gluon paged-MQA logits kernel for decode.`，trace 里 kernel 为 `_gluon_paged_mqa_logits_fused2` | `results/a1-pair1/check.txt`、`trace-cat/` |
| A1 正确性门禁 | 开：smoke 5/5、GSM8K 196/200、长上下文 32/32 + 31/32、MTP 接受长度 3.56；同时跑的原版：197/200、32/32 + 32/32、3.44。在 A/A 噪声内（之前的 A/A 是 197 / 195） | `results/a1-pair1/gate-*.out` |
| A1 decode A/B（104 请求 × 64K 输入 × 4096 输出，DP4 MTP3） | 纯 decode 窗口每步时间 p50 **50.89 ms 对 52.93 ms（−2.04 ms，−3.9%）**，均值 50.90 对 52.99，p99 52.61 对 54.50；窗口内每个 5 s 分箱都差约 2.0 ms。**单次运行**（GPU 0–3 开、4–7 关，未交换） | `results/a1-pair1/window-*.json` |
| A1 kernel 时间（引擎 trace，纯 decode 3 s，每 rank 55 步） | paged-MQA logits 每次 68.3 µs 对 138–146 µs（2.1 倍），每步 22 次，每 rank 每步 1.50 ms 对 3.04–3.21 ms；GPU 忙碌时间每步 51.3 对 53.4 ms，与 ITL 差值一致 | `results/a1-pair1/trace-cat/` |
| A1 上游 PR 改法 | 组大小改由调用方从 indexer decode metadata 传入（`decode_is_uniform ? write_max_decode_len : 0`），不再运行时取配置、也不需要环境变量（覆盖层 v3，函数级测试含拉平形式 CUDA graph 回放 PASS） | `overlay-v3/`、`results/a3-1/overlay-v3-test.txt` |
| A2 aiter AG/RS 为什么没走上 | **v0.34 的 GPU model runner 捕获 FULL graph 时 forward context 的 `batch_descriptor=None`**（`v1/worker/gpu/cudagraph_utils.py`，只有 PIECEWISE 才构造），而 `_can_use_aiter_ag_rs` 要求 `bd is not None and bd.uniform` → decode 的 FULL graph 里 MoE AG/RS 全部落到 PyNCCL（RCCL generic kernel）。aiter 侧（DP 组 `CustomAllreduce` 已创建、int32/fp32/bf16 张量、大小）都满足 | 第 3 节 |
| A2 端到端 | 进行中：修正版（FULL 下 `bd is None` 也放行）对原版；EP、EP + MoRI | `results/a2-*`（待补） |
| A3 #47872 单遍 Triton top-k | 移植完成，所有分布下结果精确；但在引擎形状（64K 上下文、104 行）下 aiter 的 radix one-block 只要 32.7 µs（引擎 trace 31.4 µs），移植版明显更慢（发射开销已排除的 graph 模式数据待补）。decode top-k 每步约 0.7 ms（1.4%），收益上限很小 | 第 4 节 |

## 1. 环境与方法

- 镜像：原版 `vllm/vllm-openai-rocm:v0.34.0`（digest `3cb773d2…`，与 `../vllm-image-v0340-20261117/Dockerfile` 的基础镜像相同），覆盖层按文件 bind mount。
  模型拷到本地 NVMe（438 GB，45 s）。不挂宿主机 libionic。
- 执行：`scripts/runner.sh` 用唯一的 `srun --jobid=295840 --overlap` 会话常驻，`scripts/q.sh` 投递，`scripts/monitor.sh` 监控（长任务结束、
  非零退出、心跳超时、作业剩余时间）。所有远程命令以 0 退出。
- 配置（与 P4D4 decode 相同）：`--data-parallel-size 4 --tensor-parallel-size 1 --kv-cache-dtype fp8_e4m3 --enable-prefix-caching
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' --max-num-seqs 64 --max-num-batched-tokens 8192 --block-size 64`，
  `VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1`（`scripts/agg_up.sh`）。未设 `--max-model-len`，v0.34 解析为
  **1,048,576**；每 rank KV 300 万 token；cudagraph `FULL_AND_PIECEWISE`，decode FULL graph 按请求数捕获（1–64）。
- 一轮 A/B（`scripts/round.sh`）：两台同时起（GPU 0–3 / 4–7）→ 各跑门禁（`scripts/gate.sh`）→ GSM8K ≥ 190/200 才 bench → 两台同时 bench。
- decode 主导 bench（`scripts/decode_bench.sh`）：104 个请求同时发出（随机 token，输入 65,536、输出 4,096，`--ignore-eos`，
  `--save-detailed`）。prefill 波约 210 s，之后约 60 s 内 104 个请求全部在 decode。`scripts/decode_window.py` 用每个请求的
  start + TTFT + 累计 ITL 还原每个 chunk 的到达时刻，只取"最后一个请求拿到首 token"到"第一个请求结束"之间的 ITL（MTP 下一个 chunk = 一步），
  并按 `trace.log` 的 start/stop 时刻剔除 profiling 时段。
- trace：`scripts/trace_when_decoding.sh` 读 `/metrics`（多 API server 时引擎不打统计日志，但多进程 Prometheus 指标按 engine 汇总），
  所有 engine `waiting=0`、`running>0` 且 prompt token 计数连续两次不变时抓 3 s torch trace。`scripts/trace_cat.py` 按类别汇总。
- 随机 token 下 MTP 接受长度只有 1.9（真实文本约 3.5），只影响每步产出 token 数，不影响每步 kernel 时间；A/B 两边相同。

## 2. A1：#47897（gfx950 gluon paged MQA logits）端到端

### 2.1 前两次无效的原因（已确认）

- 第一次：ROCm 上 MTP decode 被拉平（next_n = 1），第一版覆盖层要求 next_n ≥ 2（原 README 4.0 已更正）。
- 第二次：拉平后覆盖层用 `_gluon_lpt_group_rows()` 取组大小，内部 `get_current_vllm_config()` 在 forward 时没有配置上下文
  （v0.34 未设置时直接 raise `AssertionError`），被 `except Exception` 吞掉返回 0；`functools.cache` 让 0 永久生效 → `_gluon_group_rows` 为 0 → 全部回落 aiter。
- 本次：显式 `VLLM_ROCM_DSA_GLUON_LPT_GROUP_ROWS=4`，启动后先确认日志（`results/a1-pair1/check.txt`：aggK 1 行 gluon、aggR 0 行），
  以及 `DSA indexer decode path: use_flattening=True supports_varlen=False (next_n=4)`。

### 2.2 门禁

| 侧 | smoke | GSM8K 200 | 长上下文 needle / gsm8k（16K） | MTP 接受长度 |
|---|---|---|---|---|
| 开（GPU 0–3） | 5/5 | 196 | 32/32、31/32 | 3.56 |
| 原版（GPU 4–7，同时跑） | 5/5 | 197 | 32/32、32/32 | 3.44 |

### 2.3 decode A/B（单次运行）

| 侧 | 窗口 | 步时 p50 | 均值 | p90 | p99 |
|---|---|---|---|---|---|
| 开 | 60.5 s（剔除 trace 15 s），93,114 个 ITL | **50.89 ms** | 50.90 | 51.56 | 52.61 |
| 原版 | 59.8 s（剔除 15 s），88,080 个 ITL | **52.93 ms** | 52.99 | 53.56 | 54.50 |

窗口内 5 s 分箱中位数：开 50.1–51.0 ms，原版 52.9–53.0 ms，每个分箱都差约 2.0 ms。bench 的 TPOT（含 prefill 干扰段）不用来比较。

### 2.4 trace（每 rank 每步 ms，纯 decode 3 s 窗口，55 步）

| 侧 rank | MoE GEMM | dense GEMM | 通信 | sparse MLA | indexer logits | MoE 其他（含 decode top-k） | 合计 |
|---|---|---|---|---|---|---|---|
| 开 dp0 | 19.31 | 12.81 | 7.81 | 4.06 | **1.50** | 2.83 | 51.31 |
| 开 dp3 | 19.39 | 12.84 | 7.80 | 4.10 | **1.50** | 2.91 | 51.55 |
| 原版 dp0 | 19.66 | 12.47 | 7.65 | 4.17 | **3.21** | 2.92 | 53.20 |
| 原版 dp3 | 19.66 | 12.43 | 8.20 | 4.05 | **3.04** | 2.89 | 53.37 |

- logits kernel：`_gluon_paged_mqa_logits_fused2` 68.3 µs / 次，原版 `_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle`（aiter）138–146 µs / 次，每步 22 次。
- 本负载各 rank 上下文相同，没有 P4D4 里"最重 rank 拖慢全组"的放大；AgentX 负载下各 rank 不均，收益预计更大（原 README 4.1 的估算）。

### 2.5 上游 PR（U12）建议改法

`overlay-v3/rocm_aiter_mla_sparse.py.diff`（对 v0.34.0，kernel 文件与 `../vllm-kf-port-20261117/overlay/` 相同）：

- `rocm_fp8_paged_mqa_logits(..., flat_group_rows=0)`：拉平（next_n = 1）时由调用方给出每请求行数；
  `rocm_aiter_sparse_attn_indexer` 的 decode 段传 `decode_metadata.write_max_decode_len if decode_metadata.decode_is_uniform else 0`
  （indexer metadata builder 每步都有这两个值；FULL graph 捕获时是均匀 decode，即 1 + num_speculative_tokens）。
- 去掉 `_gluon_lpt_group_rows()` 与 `VLLM_ROCM_DSA_GLUON_LPT_GROUP_ROWS`；开关 `VLLM_ROCM_DSA_GLUON_PAGED_MQA_LOGITS` 在上游应登记到 `vllm/envs.py`
  （现在启动时会报 "Unknown vLLM environment variable"）。
- 函数级测试（`scripts/kf/test_overlay_logits.py`，自动识别 v3）：1–32 请求各种长度形式与 aiter 最大差 7.6e-6，40 请求回落 aiter，
  `[B, next_n]` 与拉平形式（`[rows, 1]` 长度）的 CUDA graph 换内容回放都一致：PASS（`results/a3-1/overlay-v3-test.txt`）。
- PR 草稿需要更新的地方：端到端数据用 2.3 / 2.4；组大小来源改成 metadata。v3 的引擎内 A/B 待补（行为应与本次相同：均匀 decode 时组为 4）。

## 3. A2：decode MoE 通信（U13）

### 3.1 aiter 自定义 AG/RS 为什么没走上（代码确认）

- DP 组（`dp:0`）的 `CudaCommunicator` 创建了 aiter `CustomAllreduce`（日志 4 个 rank 都有 "CustomAllreduce input pool"），`use_aiter_ag_rs=True`；
  MoE 分发 / 合并（`AgRsAll2AllManager.dispatch/combine`）走 `get_dp_group().all_gatherv / reduce_scatterv`，会先问 `_can_use_aiter_ag_rs(sizes)`。
- `_can_use_aiter_ag_rs` 要求：`sizes is None`（各 rank token 数相同，FULL graph padding 后成立）、`cudagraph_runtime_mode == FULL`、
  **`batch_descriptor is not None and batch_descriptor.uniform`**。
- v0.34 只有新的 GPU model runner（`v1/worker/gpu/`），它在 `cudagraph_utils.py` 捕获时：`batch_descriptor = None`，只有 PIECEWISE 才构造
  `BatchDescriptor(num_tokens=...)`（也不带 `uniform`）；FULL graph 重放时不跑 Python。所以 decode 的 FULL graph 在捕获那一刻就选了 PyNCCL，
  aiter AG/RS 在这个 runner 下是死代码。trace 里每步 156 次 `ncclDevKernel_Generic_1`（每个 MoE 层 2 次，含 MTP 层），平均约 50 µs，
  每步 7.5–8.6 ms（约 15%）。
- aiter 侧条件：`should_custom_ag` 不限 dtype（int32 按 fp32 视图）、16 字节对齐、单 rank ≤ 16 MB；`should_custom_rs` 要求行数被 world size 整除；
  本配置都满足。
- 实验修法（`overlay-comm/cuda_communicator.py.diff`）：FULL 模式下 `batch_descriptor is None` 也放行，首次放行时打印
  `Using AITER custom AG/RS for group dp:0.`。上游更合适的改法是 runner 在 FULL 捕获时也传 `BatchDescriptor(num_tokens, num_reqs, uniform=...)`，
  或者 `_can_use_aiter_ag_rs` 不再依赖 `batch_descriptor`（`sizes is None` 已保证各 rank 形状一致）。

### 3.2 端到端（进行中）

- 第一轮：GPU 0–3 修正版（`comm`）对 GPU 4–7 原版；第二轮：`--enable-expert-parallel` 对 EP + `--all2all-backend mori`。

## 4. A3：#47872（单遍 Triton top-k）/ #47276

### 4.1 #47872 移植

- `scripts/kf/vllm_triton_topk_decode.py`：sglang 的 est（Triton）→ collect（Gluon，单遍）→ select（Triton）。改动：输出行内位置而不是物理 slot
  （vLLM 的 decode top-k 输出 token 位置，sparse MLA 再查 block table；页表映射保留为 `TRANSFORM` 选项）、写入调用方的 `topk_indices`（带行 stride）、
  长度为 0 的 padding 行不除零、scratch 固定地址（CUDA graph 安全）。
- 微基准 `scripts/kf/bench_topk.py`：引擎形状（每请求 4 行，行宽 1,048,576，k = 2048），对 torch.topk 第 k 大值做平局鲁棒的精确性检查。

| 上下文 / 设置 | 行数 | aiter（引擎现用） | vLLM `_C` | #47872 移植 | 移植 / aiter |
|---|---|---|---|---|---|
| 64K，缓存热 | 104 | 32.7 µs | 85.2 | 63.5 | 0.51 倍 |
| 64K，MALL 冷 | 104 | 37.4 | 91.7 | 51.7 | 0.72 倍 |
| 8K–128K，缓存热 | 104 | 44.2 | 85.8 | 58.5 | 0.76 倍 |
| 40K–200K，MALL 冷 | 104 | 70.7 | — | 58.7 | 1.20 倍 |
| 40K–200K，大量平局 | 104 | 97.6 | — | 58.0 | 1.68 倍 |

所有组合坏行数为 0。"缓存热"的移植版数值受 Python 发射 3 个 kernel 的主机开销限制（FLUSH 时 CPU 能提前排队，反而更快），
CUDA graph 模式（引擎实际情况）的数据待补；aiter 在 64K、104 行时 32.7 µs 与引擎 trace（31.4 µs）一致。

### 4.2 #47276

待做（依赖 sglang 的 JIT HIP `dsa_gfx950` 基础设施）。

## 5. 对共享文档的更新建议（由主 session 合并）

- **U12**：状态改为"引擎内生效，decode 每步 −3.9%（单次，64K × 104 请求）"；PR 改法见 2.5；第二次 A/B 无效的原因已确认（不是推测）。
- **U13**：根因改为"v0.34 GPU model runner 的 FULL 捕获不传 `batch_descriptor`，`_can_use_aiter_ag_rs` 永远为假"；端到端数据待 3.2。
- 原 README 3.2 / 4.0 与 `bench_paged_logits.py` 用 `MAX_MODEL_LEN = 202752`，引擎实际是 1,048,576（logits 行宽 4 MB）；对 logits kernel 影响不大，
  但 top-k 等按行宽选 kernel 的地方应按 1M 测。
- `../vllm-pd-tools-20261117/scripts/trace_cat.py` 两处问题（本目录 `scripts/trace_cat.py` 已改）：
  1. 步数用 `hipGraphLaunch / 2` 估计，MTP 下每步有多个 graph，**本次估成 110 步，实际 55 步**（按每步一次的采样 kernel 计）；
     旧脚本得到的 "ms/step" 可能偏小一倍（各类占比不受影响）。
  2. 正则在 `name.lower()` 之后匹配，而 dense GEMM 的模式里是大写 `Cijk`，hipBLASLt GEMM 全部落进 "other"。
