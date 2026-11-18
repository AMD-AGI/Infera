# A 线：GLM-5.2 decode 侧 kernel / 通信（vLLM v0.34.0 ROCm，聚合模式，v3-240）

接手 `../vllm-kf-port-20261117/HANDOFF-trackA.zh-CN.md`。**交付说明（给主 session）：`HANDOFF-to-main.zh-CN.md`。**节点 v3-240（作业 295840，8×MI355X），工作目录
`/home/liyingli/bench_agentx/vllm-trackA-20261118/`（原始日志、详细 bench JSON、完整 trace 在那里，本目录只放脚本和派生结果）。

## 0. 结论摘要

**推荐的 decode 配置**：默认配置（DP4 TP1、MTP3、fp8 KV、非 EP、AITER 融合共享专家）+ #47897 覆盖层 v3 + aiter 自定义 AG/RS 修正
（`overlay-v3/`、`overlay-comm/`）。与原版同时跑（r8，64K 输入、4K 输出，纯 decode 窗口每步 p50，各一次）：

| 并发 | 推荐配置 | 原版 | 差值 |
|---|---|---|---|
| 104 请求 | **49.77 ms** | 53.12 ms | **−3.35 ms（−6.3%）** |
| 64 请求 | 41.29 | 43.62 | −2.33 ms（−5.3%） |
| 32 请求 | 34.45 | 36.38 | −1.93 ms（−5.3%） |

门禁 196/200、长上下文 32/32 + 32/32（原版 195/200）。r7 的同类组合 49.66 ms，与之一致。

| 项 | 结论 | 证据 |
|---|---|---|
| A1 #47897 前两次为何没生效 | 第二次的推测成立：覆盖层在 forward 时调 `get_current_vllm_config()`，worker 里没有配置上下文会抛异常，被吞掉后组大小为 0 并被 `functools.cache` 永久缓存 | 2.1 |
| A1 端到端 | 门禁通过（196、197、197/200）；步时 50.89 / 51.45 / 51.04 ms 对原版 52.86–52.98（5 次、两组 GPU，噪声 < 0.1 ms）→ **快 1.45–2.0 ms（2.7–3.9%）**。logits kernel 68 µs 对 138–146 µs | 2.3、2.4 |
| A1 上游改法 | 组大小由调用方从 indexer decode metadata 传入（`decode_is_uniform ? write_max_decode_len : 0`），不用环境变量；引擎内已验证（r6、r7） | 2.5、`PR-U12-…md` |
| A2 aiter AG/RS 为何没走上 | v0.34 的 GPU model runner 捕获 FULL graph 时，在 `torch.cuda.graph(...)` 里以 `cudagraph_runtime_mode=NONE`、不带 batch descriptor 跑 forward；`_can_use_aiter_ag_rs` 要求 `mode == FULL` 且 `uniform`，永远不成立 → MoE AG/RS 全走 RCCL generic kernel | 3.1 |
| A2 修正后 | FULL 捕获时放行（`mode == NONE and is_current_stream_capturing()`）：步时 **51.49 ms（−2.7%）**，门禁 196/200；通信 7.8–10.2 → 6.1 ms/步。批量 all-gatherv 无收益 | 3.2、`PR-U13-…md` |
| EP | 默认（融合共享专家）下 `--enable-expert-parallel` 结果错误（GSM8K 1/200）：DP + AG/RS 分发把共享专家的"由谁计算"标记也 gather 了。修复（`overlay-ep/`）后 196/200，但 EP 不比非 EP 快（53.40 ms；叠加其余改进 50.82 对 49.77） | 3.2 |
| MoRI 为何慢 | vLLM 接入层：接收缓冲区按 `max_num_batched_tokens`（8192）× 4 行交给 aiter，按大 M 选 kernel 并多一次量化；切片到 `num_dispatchers × 本步 token 数` 后 63.56 → 55.26 ms。combine 走 PUSH 的 `_nop2p`；创建 op 时设 `use_external_inp_buf=False` 走 `_p2p` 后再快约 0.5 ms（54.85）。修完约与 EP + RCCL AG/RS（54.49）持平，但 MoRI 不支持融合共享专家，单节点仍不如默认配置 | 3.3 |
| A3 #47872 | 移植完成、结果精确；CUDA graph 计时下 64K 上下文比 aiter 慢（36.0 对 25.3 µs），40K–200K 长上下文快 1.33 倍。只对长上下文负载有用（估每步约 0.5%） | 4.1 |
| A3 #47276 | 独立构建 sglang kernel 对比：≤ 16 行快 31–38%，P4D4 decode 形状（104 行）慢 15–32%，不移植 | 4.2 |
| 附带（U6） | 开 EP 时权重加载 14 s、起栈 151 s；不开 EP 567 s、683 s（MoE 按 TP=4 跨 DP 切分时每个 rank 读全部分片再切） | 3.2 |

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

### 2.3 decode A/B

| 侧 | 窗口 | 步时 p50 | 均值 | p90 | p99 |
|---|---|---|---|---|---|
| 开 | 60.5 s（剔除 trace 15 s），93,114 个 ITL | **50.89 ms** | 50.90 | 51.56 | 52.61 |
| 原版 | 59.8 s（剔除 15 s），88,080 个 ITL | **52.93 ms** | 52.99 | 53.56 | 54.50 |

窗口内 5 s 分箱中位数：开 50.1–51.0 ms，原版 52.9–53.0 ms，每个分箱都差约 2.0 ms。bench 的 TPOT（含 prefill 干扰段）不用来比较。

复测（A2 r6，覆盖层 v3，GPU 4–7）：门禁 197/200、长上下文 32/32 + 31/32；步时 p50 **51.45 ms（−1.45 ms，−2.7%）**。logits kernel 每次 68.0 µs、
每步 1.50 ms，与上面完全相同；差出的约 0.5 ms 分布在通信（52.0 对 50.1 µs/次）和 MoE GEMM（19.71 对 19.31 ms/步）上。原版在两组 GPU 上 5 次
都在 52.86–52.98 ms，所以 #47897 的端到端收益目前是 **1.4–2.0 ms/步（2.7–3.9%）**，r7 在 GPU 0–3 上再测一次 v3 来定位差异来源。

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
- v3 的引擎内验证：r6（GPU 4–7）51.45 ms、r7（GPU 0–3）51.04 ms，门禁 197、197/200，日志确认 gluon 生效（不设组大小环境变量）。
  PR 草稿已按此更新：`PR-U12-gfx950-gluon-paged-mqa-logits.md`。

## 3. A2：decode MoE 通信（U13）

### 3.1 aiter 自定义 AG/RS 为什么没走上（代码确认）

- DP 组（`dp:0`）的 `CudaCommunicator` 创建了 aiter `CustomAllreduce`（日志 4 个 rank 都有 "CustomAllreduce input pool"），`use_aiter_ag_rs=True`；
  MoE 分发 / 合并（`AgRsAll2AllManager.dispatch/combine`）走 `get_dp_group().all_gatherv / reduce_scatterv`，会先问 `_can_use_aiter_ag_rs(sizes)`。
- `_can_use_aiter_ag_rs` 要求：`sizes is None`（各 rank token 数相同，FULL graph padding 后成立）、`cudagraph_runtime_mode == FULL`、
  **`batch_descriptor is not None and batch_descriptor.uniform`**。
- v0.34 只有新的 GPU model runner（`v1/worker/gpu/`）。它在 `cudagraph_utils.py` 捕获 FULL graph 时传 `batch_descriptor=None`（只有 PIECEWISE
  才构造），`forward_context.set_forward_context` 再用 `BatchDescriptor(num_tokens=num_tokens)` 补上，`uniform` 为默认的 False；FULL graph 重放时
  不跑 Python。所以 decode 的 FULL graph 在捕获那一刻就选了 PyNCCL，aiter AG/RS 在这个 runner 下是死代码。trace 里每步 156 次
  `ncclDevKernel_Generic_1`（每个 MoE 层 2 次，含 MTP 层），平均约 50 µs，每步 7.5–8.6 ms（约 15%）。
- aiter 侧条件：`should_custom_ag` 不限 dtype（int32 按 fp32 视图）、16 字节对齐、单 rank ≤ 16 MB；`should_custom_rs` 要求行数被 world size 整除；
  本配置都满足。
- **更正（两次）**：最初以为只是 `batch_descriptor` 为 None / `uniform=False`，前两版实验覆盖层只放宽了 `uniform` 条件，r1、r3 都没走上 aiter。
  r4 的诊断版打印了每个早退原因：捕获期间是 `mode NONE capturing=True`——新 runner 的 `CudaGraphManager.capture` 在
  `torch.cuda.graph(...)` 里调用 `forward_fn(CUDAGraphMode.NONE)`，forward context 里根本不是 FULL，`uniform` 那一层走不到。
  调用链：`naive_dp_ep.prepare → get_ep_group().dispatch → AgRsAll2AllManager.dispatch → get_dp_group().all_gatherv → _can_use_aiter_ag_rs`。
  （r2 的诊断版因 `info_once` 参数含 list 不可哈希，在 DP 空闲 rank 的 dummy batch 里崩溃，是我的诊断代码问题。）
- 修法（r6 起生效；`overlay-comm/cuda_communicator.py` / `.diff` 是交付用的干净版，只含这处判断；实验版另带诊断日志和批量
  all-gatherv 开关，在工作目录 `overlay-comm/`）：`mode == FULL`，或 `mode == NONE` 且 `torch.cuda.is_current_stream_capturing()`
  （即 FULL 捕获）时放行；PIECEWISE 仍不放行；`sizes is None` 已保证组内各 rank 形状一致。vLLM 在 `models/deepseek_v32/attention.py`
  里已经用同样的判断（`cudagraph_runtime_mode == FULL or torch.cuda.is_current_stream_capturing()`）。上游更好的改法是 runner 在 FULL 捕获时
  把 forward context 设成 FULL（或带一个"正在捕获 FULL"的标志），否则所有"捕获时按 FULL 做保守选择"的代码都会失效，例如
  `rocm_aiter_mla_sparse.py` decode top-k 的上界（FULL 时用 `max_model_len`，否则用本步的 `max_seq_len`；GLM 的 compress ratio 1 不受影响，
  DSv4 C4A 可能在捕获时按 dummy 长度选错 kernel，未验证）。

### 3.2 端到端

| 轮次 | 配置 | 门禁 | 结果 |
|---|---|---|---|
| r1 | 第一版 `comm`（只放行 `bd is None`）/ 原版 | 198 / 199 | 两边都没走 aiter → A/A：52.98 / 52.98 ms |
| r2 | 诊断版 `comm` / `--enable-expert-parallel` | 崩溃（诊断代码 bug）/ **1/200** | EP 输出乱码（MTP 接受长度 1.03、全部 runaway），见下 |
| r3 | FULL 下直接放行的 `comm` / EP + `--all2all-backend mori` | 198 / 起不来 | `comm` 仍未走上 aiter，步时 52.89 ms（= 原版）；MoRI 断言拒绝融合共享专家 |
| r4 | 诊断版 `comm` / EP + 关闭融合（`nofuse`） | 198 / **198** | 诊断：捕获时 `mode NONE capturing=True`（根因见 3.1）；`comm` 52.86 ms（= 原版）。EP + 关闭融合正确，步时 **54.49 ms** |
| r5 | EP + MoRI + 关闭融合 / 非 EP + 关闭融合 | 194 / 195 | EP + MoRI **63.56 ms**；非 EP 关闭融合 **55.33 ms** |
| r6 | FULL 捕获放行的 `comm` / 覆盖层 v3（GPU 4–7） | **196** / 197 | **aiter AG/RS 生效**（`Using AITER custom AG/RS for group dp:0.`，三个分发张量与 RS 都满足），步时 **51.49 ms（−1.4 ms，−2.7%）**；v3 51.45 ms（见 2.3） |

aiter AG/RS 生效后的 trace（dp0，每步）：通信 6.10 ms（原版 7.8–10.2）。hidden states 的 AG（`aiter::allgather_vec<bf16>`）每次 34.5 µs、
RS（`reduce_scatter_split_first_dim`）34.0 µs，替代约 50–65 µs 的 RCCL generic；top-k weights / ids 各一次 `allgather_vec<float>`（4.8 µs）。
aiter 有 `custom_all_gatherv`（一次 launch、一次 rank 屏障收最多 4 个张量），覆盖层加了开关 `VLLM_ROCM_AITER_AG_BATCHED=1`，r7 测。

| r7 | 覆盖层 v3（GPU 0–3）/ **组合**：v3 + aiter AG/RS + 批量 all-gatherv（GPU 4–7） | 197 / **199** | v3 **51.04 ms**；组合 **49.66 ms（−3.25 ms，−6.1%）** |

- 组合 ≈ 两项相加（#47897 约 1.45–2.0 ms + aiter AG/RS 约 1.4 ms）。
- 批量 all-gatherv 没有收益：`aiter::allgatherv<4>` 每次 48.0 µs，比分开的 34.5 + 2×4.8 = 44.1 µs 还慢（通信 6.54 对 6.10 ms/步，单次）。不建议打开。
| r8 | **推荐配置**：v3 + aiter AG/RS（不批量）/ 原版，同时跑，之后同一对服务测 64、32 请求 | 196 / 195 | 见下表 |
| r9 | EP + 融合共享专家 + **修复**（其余原版）/ EP + 修复 + v3 + aiter AG/RS | 196 / 197 | 修复后 EP 结果正确（未修复是 1/200）；步时 53.40 / 50.82 ms，都不比非 EP 快 |

r8 最终 A/B（64K 输入、4K 输出，纯 decode 窗口每步 p50，各一次）：

| 并发 | 推荐配置 | 原版 | 差值 |
|---|---|---|---|
| 104 请求 | **49.77 ms** | 53.12 ms | **−3.35 ms（−6.3%）** |
| 64 请求 | 41.29 | 43.62 | −2.33 ms（−5.3%） |
| 32 请求 | 34.45 | 36.38 | −1.93 ms（−5.3%） |

104 请求的结果与 r7 组合版（49.66 ms，含批量 all-gatherv）一致。门禁：推荐配置 196/200、长上下文 32/32 + 32/32；原版 195/200。

- EP 修复（r9）：门禁 196/200、长上下文 32/32 + 31/32（叠加版 197/200、32/32 + 32/32），正确性恢复；但 EP + 融合 53.40 ms ≈ 默认（52.9–53.1），
  叠加 v3 + aiter AG/RS 后 50.82 ms，比非 EP 的推荐配置（49.77）慢约 1 ms。所以本负载下**不建议 EP**；修复本身作为正确性修复提给上游。
  （之前按"关闭融合时 EP 比非 EP 快 0.84 ms"推算 EP + 融合会更快，实测不成立。）

各配置每步时间（p50，纯 decode 窗口，各一次运行；默认配置有 4 次在 52.86–52.98 ms）：

| 配置 | 步时 | 相对默认 | trace（dp0，每步）通信 / MoE GEMM |
|---|---|---|---|
| 默认：非 EP、融合共享专家、AG/RS（RCCL） | 52.9 ms | — | 7.8–10.2 / 19.3–19.6 ms |
| 非 EP、关闭融合 | 55.33 | +2.4 | 8.0 / 20.1 |
| EP（AG/RS）、关闭融合 | 54.49 | +1.6 | 8.1 / 19.0 |
| EP + MoRI（EPv1）、关闭融合 | 63.56 | +10.7 | dispatch 35 µs + combine 70 µs 每次 / 24.0 |

- 共享专家融合值约 2.4 ms/步；同样关闭融合时 EP 比非 EP 快 0.84 ms，但 EP 现在必须关闭融合（否则结果错误），净值更慢。
  修好 EP + 融合后，预计可在默认基础上再省约 0.8 ms（推算，未测）。
- MoRI 在本节点用的是 `EpDispatchIntraNodeKernel_fp4` / `EpCombineIntraNodeKernel_bf16_nop2p`（**nop2p**，未走 GPU 直连），
  combine 每次 70 µs，MoE GEMM 也变慢；单节点 decode 不建议用。nop2p 的原因（容器内 P2P 未启用还是 MoRI 1.2.3 的限制）待查。

- **EP 正确性问题**：默认的 `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1` 下开 `--enable-expert-parallel`，每个 rank 是 65/256 个本地专家
  （64 个路由专家 + 融合进来的共享专家），GSM8K 1/200。MoRI 后端对同一组合直接断言
  `Mori does not support fusion shared expert now. Turn it off by setting VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0`，
  而 AG/RS 后端没有这个检查。r4 关闭融合（`=0`）后 EP 门禁 198/200，确认是融合共享专家与 EP 的组合出错。
  机制（读代码得出）：EP 下融合进来的共享专家（全局 id 256）在**每个** rank 都映射到本地槽位（所以是 65/256）。为了让每个 token 的共享专家
  只算一次，`rocm_aiter_moe.init_aiter_topK_meta_data(is_EP=True)` 在 meta 缓冲区（行数按 `max_num_batched_tokens × dp_size` 分配）里
  只给 `row % ep_size == ep_rank` 的行填真实的共享专家 id，其余行填被 `expert_mask` 屏蔽的哨兵 id。这在各 rank token 相同（TP+EP）时成立；
  但 DP + AG/RS 分发时，top-k ids **连同共享列**一起从源 rank gather 过来，每行带的是源 rank 按其本地行号做的标记——
  被标记的 1/4 token 在 4 个 rank 上各算一次（×4），其余 3/4 一次都不算。
- 修法（`overlay-ep/`，r9 验证）：`naive_dp_ep.prepare` 在 dispatch 之后用本 rank meta 缓冲区的共享列覆盖 gather 结果的共享列
  （按 gather 后的行号，各 rank 各算 1/4，负载均衡）；`rocm_aiter_moe.py` 记录共享列宽度和是否 EP（非 EP 跳过）。
  上游应修这一处（或让 AG/RS 后端像 MoRI 一样拒绝该组合）。
- MoRI 1.2.3.post1 低于 EPv2 要求的 1.2.4，vLLM 回落 EPv1。
- r5：EP + MoRI + 关闭融合 / 非 EP + 关闭融合（EP 的同口径基线）；r6：本次修正的 `comm` / 覆盖层 v3（A1 在 GPU 4–7 上复测）。

### 3.3 MoRI 为什么慢（r5 trace + 读 mori 1.2.3 源码）

EP + MoRI（EPv1，IntraNode，关闭融合）每步 63.56 ms，比 EP + AG/RS（54.49）多约 9 ms，来自三处（dp0 trace，`scripts/trace_kernels.py`）：

| 项 | EP + MoRI | EP + AG/RS |
|---|---|---|
| moe1 | `…bf16_t64x128x256…` 每次 190 µs | `…bf16_t32x128x256…` 138 µs |
| `dynamic_per_group_scaled_quant<bf16→fp4>` | 每层 2 次：p10 4.3 µs（dispatch 前，104 token）+ **56 µs** | 每层 1 次 5.0 µs |
| 分发 / 合并 | `EpDispatchIntraNodeKernel_fp4` 35 µs + `EpCombineIntraNodeKernel_bf16_nop2p` 70 µs | RCCL AG / RS 各约 52 µs |

- **`_nop2p` 不是硬件 P2P 不可用**：mori 按 `useExternalInpBuffer` 选 combine kernel（`launch.cpp`：true → `_nop2p` = PUSH，false → `_p2p` = PULL / 零拷贝）。
  vLLM 的 `MoriAll2AllManager._make_all2all_kwargs` 没设 `use_external_inp_buf`（mori 默认 True）；零拷贝需要 expert 把输出写进
  `EpDispatchCombineOp.get_registered_combine_input_buffer()`，vLLM 的 `MoriPrepareAndFinalize` 没有实现（`allocate_fused_expert_output` 这个钩子可以做）。
- **接收容量**：`max_num_inp_token_per_rank = moe.max_num_tokens = max_num_batched_tokens`（8192），也没设 `max_total_recv_tokens`，
  接收缓冲区按 8192×4 行分配；dispatch 返回的就是这个容量的张量（实际行数在设备端的 `dispatch_recv_token_num` 里），
  aiter `fused_moe` 按这个 M 选了大 tile、不融合量化的 moe1 变体，中间量化也在补齐后的缓冲区上做（56 µs）。
- r10 验证容量：`--max-num-batched-tokens 1024`（容量缩小 8 倍），EP + MoRI 对 EP + AG/RS，两边同样参数。prefill 变慢后不存在 104 个请求
  同时 decode 的窗口，ITL 指标算不出来；用纯 decode 时的 trace 比较（dp0，每步）：

| 项 | MoRI 8192（r5） | MoRI 1024（r10） | EP + AG/RS 1024（r10） |
|---|---|---|---|
| 中间量化（第二次） | 56 µs | **8.8 µs** | — |
| moe1 | t64，190 µs | t128，161 µs | t32，138 µs |
| MoE GEMM 合计 | 23.98 ms | 22.61 ms | 18.89 ms |
| 分发 + 合并 | 约 8 ms | 约 7 ms（combine `_nop2p` 62 µs） | 7.71 ms（RCCL） |
| GPU 忙碌 / 步 | 66.9 ms | **59.3 ms** | 53.5 ms |

  容量缩小后 MoRI 每步省约 7.6 ms，通信已与 RCCL AG/RS 持平；剩下约 5.8 ms 主要是 MoE GEMM：aiter 拿到的 M 仍是接收缓冲区行数（4096），
  而不是实际最多 4 × 104 = 416 行，选了大 tile。
- 实验修法（`overlay-mori/`）：
  1. `VLLM_MORI_SLICE_RECV=1`：`prepare` 把接收缓冲区切成 `num_dispatchers × 本步各 rank 最大 token 数` 行再交给 expert（MoRI 的接收槽位由原子
     fetch_add 分配、紧凑排在前面，这个上界在 CUDA graph 下也成立），不必降 `max_num_batched_tokens`；
  2. `VLLM_MORI_ZERO_COPY_COMBINE=1`：`allocate_fused_expert_output` 返回 `get_registered_combine_input_buffer()`，`finalize` 以
     `use_external_inp_buf=0` 调 combine（`_p2p` / PULL）。
  r11：零拷贝 × {1024, 8192}；r12：切片、切片 + 零拷贝（8192）。

| 配置（EP、关闭融合、`max_num_batched_tokens` 8192） | 门禁 | 步时 p50 |
|---|---|---|
| MoRI 原样（r5） | 194 | 63.56 ms |
| MoRI + 零拷贝（按调用覆盖，r11） | 197 | 64.02 |
| **MoRI + 接收缓冲区切片**（r12） | 197 | **55.26（−8.3 ms）** |
| MoRI + 切片 + 零拷贝（按调用覆盖，r12） | 196 | 55.19 |
| 对照：EP + AG/RS（r4） | 198 | 54.49 |

- **主因是接收缓冲区容量**：切片后 moe1 回到 t32 tile（139.7 µs，与 AG/RS 相同），第二次量化消失，MoE GEMM 24.3 → 20.8 ms/步。
  MoRI 与 EP + AG/RS 只差约 0.7 ms；剩下的主要是 combine（`_nop2p` 每层 66 µs）。
- r11/r12 的"零拷贝"实际没生效：trace 里 combine 仍是 `_nop2p`。mori 1.2.3 的 `combine(use_external_inp_buf=0)` 按调用覆盖不改变
  C++ 侧（`launch.cpp` 按创建时的 `config.useExternalInpBuffer`）选的 kernel。r13 改为创建 op 时就设 `use_external_inp_buf=False`
  （`overlay-mori/…/all2all.py`），并在 `finalize` 中保证 combine 读到的是注册缓冲区。
- r13（同时跑）：切片 + 配置级零拷贝 **54.85 ms**（门禁 197），combine 变为 `EpCombineIntraNodeKernel_bf16_p2p`，每次 59.6 µs（`_nop2p` 68.3）；
  只切片的复测 55.33 ms（r12 为 55.26）。零拷贝时 expert 输出并不总在注册缓冲区上（`finalize` 里兜底拷贝了，每层多一次约 4 µs 的 copy），
  modular kernel 里还有一处会换掉输出别名，未继续追。
- 结论：MoRI 修完（切片 + 零拷贝）约 54.5–54.9 ms，与 EP + RCCL AG/RS 持平；它目前不支持融合共享专家（约 2.4 ms/步的收益），
  单节点 DP4 decode 不如默认的非 EP 配置。MoRI 的价值主要在多节点 EP（P8D8 的跨节点 all2all），不在本节点范围。
  建议给 vLLM 的改动：MoRI `prepare` 里把接收缓冲区切到 `num_dispatchers × max(num_tokens_across_dp)` 行再交给 expert；
  IntraNode 时以 `use_external_inp_buf=False` 创建 op 并让 expert 直接写注册缓冲区。

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

所有组合坏行数为 0。上表是逐次计时，移植版受 Python 发射 3 个 kernel 的主机开销限制（FLUSH 时 CPU 能提前排队，反而更快），
不代表引擎里的情况。下表是 CUDA graph 重放计时（`GRAPH=1`，缓存热、无发射开销，即引擎 FULL decode graph 的情况，`results/a3-2/`）：

| 上下文 | 16 行 | 32 | 64 | 104 | 128 |
|---|---|---|---|---|---|
| 64K（aiter / 移植，µs） | 27.9 / 29.7 | 30.2 / 31.0 | 24.7 / 33.3 | **25.3 / 36.0（0.70 倍）** | 25.8 / 40.1 |
| 8K–128K | 34.3 / 29.9 | 37.1 / 32.9 | 37.6 / 33.1 | 37.2 / 36.7（1.02 倍） | 39.3 / 39.9 |
| 40K–200K | 33.2 / 30.7 | 41.6 / 30.9 | 51.8 / 34.6 | **54.4 / 40.9（1.33 倍）** | 55.2 / 43.2 |

- aiter 的 radix one-block 在上下文短而整齐时很快（引擎 trace 31.4 µs），长上下文时随长度增长；移植版的单遍 collect 对长度不敏感。
- FULL graph 不按上下文长度区分，不能按长度选 kernel；只有在已知长上下文为主的负载（AgentX decode）上才值得默认打开。
  P4D4 s6c 的 top-k 是每次约 47 µs（上下文比本 bench 长），按 40K–200K 的比例每步约省 0.3 ms（约 0.5%），需要长上下文负载的端到端 A/B 才能确认，优先级低。

### 4.2 #47276（融合 indexer logits + 直方图 + top-k）

- sglang 的 `dsa_gfx950` kernel 依赖其 JIT（tvm-ffi + `sgl_kernel` 头文件）。镜像里有 hipcc、ninja、`apache-tvm-ffi 0.1.11`，
  `scripts/kf/bench_sgl47276.py` 按 sglang `load_jit` 的方式生成包装 `.cu`（`TVM_FFI_DLL_EXPORT_TYPED_FUNC(…, Kernel::run)`），
  hipcc 编译（加 `-DUSE_ROCM -DHIP_FP8_TYPE_E4M3=1`），`tvm_ffi.load_module` 加载，直接传 torch 张量。源码取自 PR head `3d739b80`。
- 页布局与 vLLM 的 64 token 预洗牌 indexer cache 相同（每页 64×128 FP8 K 后接 64 个 f32 scale，8448 字节），logits 与 gluon 最大差 7.6e-6；
  top-k 经页表映射，传恒等页表即得 vLLM 需要的行内位置，结果精确。
- 分组 kernel（variant 2）按每请求 `HOST_GR` 行编译（sglang 为 6，即 MTP5），只支持 ≤ 64 行；这里改成 4（MTP3）重编。

CUDA graph 计时，vLLM = gluon logits（#47897，4 行组）+ aiter top-k，sgl = `logits_hist`（按 sglang 的 `logits_launch` 规则选 variant）+ `topk_transform`：

| 行数 | 64K：vLLM / sgl 合计 µs | 比值 | 100K–128K：vLLM / sgl | 比值 | sgl variant |
|---|---|---|---|---|---|
| 16 | 42.3 / 29.2 | **0.69** | 60.0 / 37.2 | **0.62** | 1（短批） |
| 32 | 54.7 / 60.4 | 1.10 | 73.5 / 88.7 | 1.21 | 2（分组，4 行） |
| 64 | 66.1 / 109.2 | 1.65 | 103.7 / 180.1 | 1.74 | 2 |
| 104 | 93.0 / 118.6 | 1.28 | 158.7 / 182.8 | 1.15 | 0（逐行） |
| 128 | 115.5 / 152.2 | 1.32 | 180.0 / 217.8 | 1.21 | 0 |

- sgl 的优势在 top-k：用 logits kernel 顺带算出的直方图选阈值，≤ 64 行时只要 12–21 µs（aiter 28–44 µs）；logits 本身比 #47897 慢。
- 结论：P4D4 decode 形状（每 rank 约 26 请求、104 行）下不移植。低并发（每 rank ≤ 4 请求）时"短批 logits + 直方图 top-k"可让 indexer 快约 1/3，
  若以后要优化低并发延迟，可只移植 variant 1 + `topk_transform`，按行数（捕获时已知）选择。

## 5. 对共享文档的更新建议（由主 session 合并）

- **U12**（#47897）：状态改为"引擎内生效，decode 每步 −2.7% 至 −3.9%（3 次，64K × 104 请求），门禁通过"；改法用覆盖层 v3（组大小来自 metadata）；
  第二次 A/B 无效的原因已确认。PR 草稿：`PR-U12-gfx950-gluon-paged-mqa-logits.md`。
- **U13**（MoE 通信）：根因改为"v0.34 GPU model runner 以 `cudagraph_runtime_mode=NONE` 捕获 FULL graph，`_can_use_aiter_ag_rs` 永远为假"；
  修正后 −2.7%，与 U12 合起来 −6.3%。PR 草稿：`PR-U13-aiter-ag-rs-full-capture.md`。同一问题也影响其他"捕获时按 FULL 做保守选择"的代码。
- **新候选**：
  1. EP + AITER 融合共享专家 + AG/RS 分发结果错误（`overlay-ep/fused-shared-expert-ep.diff`）；
  2. MoRI 接入：接收缓冲区切片、IntraNode 零拷贝 combine（`overlay-mori/mori-slice-zerocopy.diff`）；
  3. 开 EP 时起栈快 4 倍以上（U6 的一个方向，需确认不开 EP 时能否只读本 rank 需要的切片）。
- P4D4 decode 建议：用推荐配置（非 EP）；不要开 EP 或 MoRI。PD 端到端（真实文本，MTP 接受长度约 3.5）的收益待主 session 测。
- 原 README 3.2 / 4.0 与 `bench_paged_logits.py` 用 `MAX_MODEL_LEN = 202752`，引擎实际是 1,048,576（logits 行宽 4 MB）；对 logits kernel 影响不大，
  但 top-k 等按行宽选 kernel 的地方应按 1M 测。
- `../vllm-pd-tools-20261117/scripts/trace_cat.py` 两处问题（本目录 `scripts/trace_cat.py` 已改）：
  1. 步数用 `hipGraphLaunch / 2` 估计，MTP 下每步有多个 graph，**本次估成 110 步，实际 55 步**（按每步一次的采样 kernel 计）；
     旧脚本得到的 "ms/step" 可能偏小一倍（各类占比不受影响）。
  2. 正则在 `name.lower()` 之后匹配，而 dense GEMM 的模式里是大写 `Cijk`，hipBLASLt GEMM 全部落进 "other"。
- 多 API server（DP4 默认 4 个）时引擎不打印统计日志行；`/metrics` 是多进程 Prometheus、按 engine 标签汇总，可以用（本目录
  `scripts/trace_when_decoding.sh`）。
