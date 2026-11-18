# A 线交付说明（给主 session，2026-11-18 15:10）

A 线（decode 侧 kernel / 通信，v3-240，作业 295840，04:13–14:00）已结束。详细数据和推理在同目录 `README.zh-CN.md`（第 0 节摘要、第 5 节建议），
本文件只列需要主 session 做的事。原始日志、详细 bench JSON、完整 trace 在 `/home/liyingli/bench_agentx/vllm-trackA-20261118/`（NFS，可读）。

## 1. 结论（一句话）

P4D4 的 decode 引擎用**默认配置 + #47897 覆盖层 v3 + aiter AG/RS 修正**：与原版同时跑，104 请求 × 64K 输入 × 4K 输出时纯 decode 每步 49.77 对 53.12 ms
（−6.3%），64 / 32 请求各 −5.3%，门禁 196/200、长上下文 32/32 + 32/32（原版 195/200）。不要开 EP，不要用 MoRI。

## 2. 在 P4D4 decode 上启用推荐配置

只对 decode 引擎生效（prefill 的 PP4 没有 DP 组，也不走 decode indexer 路径），三个文件按文件 bind mount 到镜像的
`/usr/local/lib/python3.12/dist-packages/vllm/`（`V` 代表这个路径）：

| 宿主机文件（仓库内） | 挂到 | 作用 |
|---|---|---|
| `llying/vllm-decode-trackA-20261118/overlay-v3/rocm_aiter_mla_sparse.py` | `$V/v1/attention/ops/rocm_aiter_mla_sparse.py` | #47897 接入（v3：组大小来自 indexer metadata） |
| `llying/vllm-kf-port-20261117/overlay/vllm/v1/attention/ops/rocm_gluon_paged_mqa_logits.py` | `$V/v1/attention/ops/rocm_gluon_paged_mqa_logits.py` | #47897 kernel 文件 |
| `llying/vllm-decode-trackA-20261118/overlay-comm/cuda_communicator.py` | `$V/distributed/device_communicators/cuda_communicator.py` | aiter AG/RS 在 FULL 捕获时放行 |

环境变量：`VLLM_ROCM_DSA_GLUON_PAGED_MQA_LOGITS=1`（v3 不需要 `VLLM_ROCM_DSA_GLUON_LPT_GROUP_ROWS`）。其余配置不变（`VLLM_ROCM_USE_AITER=1`、
`VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1`、DP4 TP1、MTP3、fp8 KV、block 64）。

- 三个文件都是对 v0.34.0 原文件的整文件替换。`infera-vllm:v0340-*` 镜像没有打 vLLM 源码补丁，应当相同；挂之前请先确认原文件一致，例如
  `docker run --rm --entrypoint md5sum <IMAGE> $V/distributed/device_communicators/cuda_communicator.py`，与 `vllm/vllm-openai-rocm:v0.34.0` 中的对比。
- 起来后确认 decode 引擎日志里有这两行，没有就说明没生效：
  `Using the gfx950 gluon paged-MQA logits kernel for decode.`、`Using AITER custom AG/RS for group dp:0.`
- `overlay-comm/cuda_communicator.py` 是交付用的干净版：只改 `_can_use_aiter_ag_rs` 的判断，行为与 r6–r8 测过的实验版相同，去掉了诊断日志和批量
  all-gatherv 开关（后者无收益）。这个干净版本身没有单独上机跑过；第一次用时请看上面那行日志并过一遍门禁。
- `scripts/agg_up.sh` 的 `gluon3+comm` 模式就是这套挂载（路径指向工作目录，复用时改 `R`）。

## 3. 需要更新的共享文件

### 3.1 `upstream-vllm-20261117/CANDIDATES.zh-CN.md`

`37fbb41e6` 合并的是 A 线较早的版本，U12 / U13 已过时（U13 的根因是我中途写错、之后更正的说法）。建议替换为：

```
| U6 | **DP 下 decode 起栈约 11 min**：每个 DP rank 读全部 282 个 shard（约 2 s / shard），超过默认引擎就绪超时 600 s。对照：开 `--enable-expert-parallel` 时每个 rank 只读本地 65/256 个专家，权重加载 14 s、起栈 151 s（不开 EP 567 s、683 s） | `../vllm-agg-correctness-20261117/` 3 节；`../vllm-decode-trackA-20261118/` 3.2 | DP rank 只读自己需要的 MoE 切片（EP 已证明可行）；默认超时随模型大小放宽 | vLLM | 待量化 |
| U12 | **gfx950 paged MQA logits（移植 sglang #47897）**：引擎内 logits 每次 68 µs 对 138–146 µs（2.1 倍）；DP4 MTP3 纯 decode 每步快 1.45–2.0 ms（−2.7% 至 −3.9%，3 次，原版 6 次 52.86–53.12 ms），门禁正常 | `../vllm-decode-trackA-20261118/` 2 节、`PR-U12-…md` | 覆盖层 v3：组大小由调用方从 indexer decode metadata 传入（`decode_is_uniform ? write_max_decode_len : 0`），不在运行时取配置 | vLLM | PR 草稿已更新，待提 |
| U13 | **DP decode 的 MoE AG/RS 走 RCCL 通用 kernel**：v0.34 的 GPU model runner 在 `torch.cuda.graph(...)` 里以 `cudagraph_runtime_mode=NONE` 捕获 FULL graph，`_can_use_aiter_ag_rs` 要求 `mode == FULL`（和 `uniform`），永远为假，aiter 自定义 AG/RS 是死代码（每步约 156 次 RCCL、约 15%） | `../vllm-decode-trackA-20261118/` 3.1–3.2、`PR-U13-…md` | FULL 捕获（`mode == NONE and torch.cuda.is_current_stream_capturing()`）时放行：每步 −2.7%；与 U12 合起来 −6.3%。更好的改法是 runner 捕获时把 forward context 设成 FULL | vLLM | PR 草稿已写，待提 |
| U17 | **EP + AITER 融合共享专家 + AG/RS 分发结果错误**（GSM8K 1/200）：`init_aiter_topK_meta_data(is_EP=True)` 按行号给 1/ep_size 的行标记共享专家，AG/RS 分发把源 rank 的标记一起 gather，1/4 的 token 共享专家算 4 次、其余 0 次；MoRI 后端对该组合断言拒绝，AG/RS 没有检查 | `../vllm-decode-trackA-20261118/` 3.2、`overlay-ep/` | dispatch 后用本 rank meta 缓冲区的共享列覆盖 gather 结果（已验证 196/200）；或像 MoRI 一样拒绝 | vLLM | 待提（本配置不开 EP，不影响 P4D4） |
| U18 | **MoRI（EPv1 IntraNode）接入慢**：接收缓冲区按 `max_num_batched_tokens × world` 行整块交给 aiter，按大 M 选 kernel、多一次量化（63.56 ms/步）；combine 用 PUSH 的 `_nop2p` | `../vllm-decode-trackA-20261118/` 3.3、`overlay-mori/` | prepare 里切到 `num_dispatchers × max(num_tokens_across_dp)` 行（55.26 ms）；IntraNode 以 `use_external_inp_buf=False` 创建 op 走 `_p2p`（54.85 ms）。修完与 EP + RCCL 持平，单节点仍不如非 EP | vLLM | 待提（多节点 EP 时更相关） |
```

### 3.2 `vllm-kf-port-20261117/README.zh-CN.md`

- 4.0b：第二次 A/B 无效的原因已确认（不是推测），见 A 线 README 2.1。
- 3.2 / 4.0 与 `bench_paged_logits.py` 用 `MAX_MODEL_LEN = 202752`，引擎实际解析为 1,048,576（logits 行宽 4 MB）。
- 第 5 节"下一步"：A1 已完成；#47872 只在长上下文时有收益（CUDA graph 计时 64K 慢 0.70 倍、40K–200K 快 1.33 倍）；#47276 在 104 行时慢 15–32%，不移植。

### 3.3 `vllm-pd-tools-20261117/scripts/`

- `trace_cat.py`：步数按 `hipGraphLaunch / 2` 估计，MTP 下估成两倍（110 对实际 55），"ms/step" 偏小一倍；dense GEMM 正则里的大写 `Cijk` 在
  `lower()` 后匹配不上。修好的版本：A 线 `scripts/trace_cat.py`（`--steps-kernel`，默认按采样 kernel 计步）。之前用旧脚本得出的每步毫秒数需要按此复核。
- 多 API server 时引擎不打统计日志行，按日志找"纯 decode"时刻的脚本不会触发；`/metrics` 是按 engine 汇总的，可用 A 线 `scripts/trace_when_decoding.sh`。

### 3.4 `vllm-pd-progress-20261117/`

A 线状态：完成。推荐配置与数据见本文第 1、2 节。

## 4. 留给主 session 的事

1. 在 P4D4（PD、真实文本，MTP 接受长度约 3.5）上测推荐配置的端到端收益；A 线的 bench 是随机 token（接受长度约 1.9）、聚合模式。
2. 推荐配置的再一次复测（A 线 r7、r8 两次：49.66、49.77 ms）。
3. 长上下文负载下 #47872 的引擎内 A/B（预计每步约 0.5%，优先级低；移植文件 `scripts/kf/vllm_triton_topk_decode.py`，还没接进
   `rocm_aiter_mla_sparse.py`）。
4. 提 U12 / U13 的上游 PR：kernel 文件保留 AMD 的 MIT 头并注明 sglang #47897，许可处理需要确认；U17 / U18 可作为 issue 或单独 PR。

## 5. 文件索引

| 内容 | 位置 |
|---|---|
| 全部结论、数据、推理 | `README.zh-CN.md` |
| PR 草稿 | `PR-U12-gfx950-gluon-paged-mqa-logits.md`、`PR-U13-aiter-ag-rs-full-capture.md` |
| 覆盖层 / diff | `overlay-v3/`、`overlay-comm/`、`overlay-ep/`、`overlay-mori/` |
| 每轮结果（门禁、decode 窗口、trace 分解） | `results/a1-pair1/`、`results/a2-r1/` … `results/a2-r13/`、`results/a3-1/`、`results/a3-2/` |
| 脚本（runner、起服务、A/B 轮次、decode 窗口、trace） | `scripts/`（`round.sh`、`final_ab.sh`、`decode_bench.sh`、`decode_window.py`、`trace_kernels.py`、`kf/`） |
