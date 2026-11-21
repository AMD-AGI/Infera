# Infera#235 的 recipe 里，哪些可以不打补丁加进 InferenceX#5138（11-19）

问题：[AMD-AGI/Infera#235](https://github.com/AMD-AGI/Infera/pull/235)（`examples/sglang_pd_glm5.2_agentx/`，GLM-5.2 PD 的 AgentX 调优 kit）里，
哪些 feature 能加进 [SemiAnalysisAI/InferenceX#5138](https://github.com/SemiAnalysisAI/InferenceX/pull/5138)
（`glm5.2-fp4-mi355x-infera-sglang-agentic-disagg`）提升性能。约束：feature 必须在 #5138 现有的 SGLang 镜像里已经可用，
不能靠打补丁开启。

#5138 的镜像：`lmsysorg/sglang-rocm:v0.5.23-rocm720-mi35x-20261117`（digest `sha256:e016f334…`，SGLang `703e763d`）。
以下"stock 里有 / 没有"都是在 `703e763d` 的源码里逐项查的；PR 状态截至 11-19 09:00 UTC。

11-21 补充：crusoe 上的 A/B 实测见 [`crusoe-ab-20261121.zh-CN.md`](crusoe-ab-20261121.zh-CN.md)。d4 上 K3 没有吞吐收益，mem 0.95 显存不够，
router 400k 在 d4 C64 +2.2%，stream-interval + CLIP 没测；2P1D / 1P1D 在 stock Mooncake 上都要加 `MC_RDMA_NOTIFY_ENABLED=0`。

## 1. 结论

#5138 的 recipe 与 Infera 最早的基线 kit（[#210](https://github.com/AMD-AGI/Infera/pull/210) 的 `examples/sglang_2p1d_glm5.2`）逐项相同
（`index_share_for_mtp_iteration: false`、两侧 `SGLANG_OPT_USE_TOPK_V2=true`、EAGLE 5/1/6、decode mem 0.92、两侧 `--disable-custom-all-reduce`），
所以要看的只是 #235 在这个基线上新增了什么。

不打补丁就能开、并且有实测收益的四项，按收益大小排：

| # | 改动 | 实测（Infera 4+4，forced acceptance） | 前提 / 风险 |
|---|---|---|---|
| 1 | 高并发点 MTP K5 → K3（`speculative-num-steps 3`、`speculative-num-draft-tokens 4`，prefill / decode 两侧都改） | C96 +4.8%（llr27：step p50 55.3 → 44.6 ms，verify graph 显存 10.5 → 7.5 GB）；C64 反而 K5 高 1.9%（42,356 vs 41,557） | 只在 decode 每 rank 跑到约 14–16 个请求以上时划算；InferenceX 按 K 自动注入 golden AL（K3 2.99，K5 3.61） |
| 2 | decode `--mem-fraction-static` 0.92 → 0.95 | K3 之后 +1.1%（llr33：每 rank KV 2.92M → 3.04M tokens，prealloc-wait 0.44 → 0.36）；无 custom AR、无 KF、K5 时 0.95 能跑但结果不确定（llr2：qps +2.4%，ISL −6%） | 开 custom AR + KF + K5 时 0.95 报 `HSA_STATUS_ERROR_OUT_OF_RESOURCES`（llr8；#235 的 K5 C88 也 3/3 中止）。#5138 没有 custom AR、没有 KF，余量更大，但仍需盯 OOM |
| 3 | 4 卡 decode 形状的 router `decode-request-weight: 400000` | C104（K3，单机 4+4）：52,813 vs 53,209 tok/s/GPU（−0.7%），interactivity p90 61.0 → 65.0（+6.6%）；#235 报告 C32 / C48 / C64 / C88 / C104 每个点都往右上移 | 这是 Infera router 参数，不是 SGLang；v0.3.3 已支持，#5138 的 `override_1p1d` 已经在用 1,000,000。代价是 TTFT p90（C88 +23%，C104 +2.5%） |
| 4 | decode `--stream-interval 30` + `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=8` | +0.9%（llr13）、+0.35%（llr14），合计约 +1%，在 ±1% 噪声内 | 首 token 仍立即发出，TTFT 不受影响；InferenceX 不少 SGLang agentic recipe 已用 `stream-interval` 20–50 |

能开但不建议加：

- decode `SGLANG_OPT_USE_TOPK_V2=false`：只在配合 KF overlay 的 top-k 时有收益，不用 KF 时 true 更好（3.6 节）。
- `index_share_for_mtp_iteration: true`（checkpoint 原生值）：两次实测一正一负（4+4 C80 +0.9%，mixed −1.3%），都在噪声内，
  #210 因此选了 false；stock 镜像还有空闲 DP rank 访存错误（MAF）的风险（3.5 节）。

开不了（要补丁 / overlay，或违反 InferenceX 规则）：decode custom all-reduce 与 `SGLANG_USE_ROCM700A=0`、prefill `--pp-async-batch-depth 1` 与
`SGLANG_PP_COMM_OVERLAP=1`、evict headroom、HiCache backup quota、KF kernels、aiter tuned GEMM CSV、FP8 draft MoE、Mooncake 的
`MC_ENABLE_DEST_LOCAL_RAIL` / `MC_IB_TIMEOUT`（第 4 节）。

上面的收益都是在 Infera 的完整栈（KF overlay、custom AR、FP8 draft、三个 prefill 补丁）上一次改一个变量测出来的，单次运行，噪声约 ±1%；
在 #5138 的 stock 镜像上幅度会不同，每项都要在 InferenceX 路径上扫过才能定。

## 2. 按 #5138 的三个形状落地

| 形状（#5138 variant） | conc | decode 每 rank 请求数 | 建议 |
|---|---|---|---|
| `override_2p1d`（2 × PP4 prefill + TP8/DP8 decode，16 卡） | 176 | 约 22 | K3 + decode mem 0.95 |
| `override_1p1d_d4`（PP4 prefill + TP4/DP4 decode，8 卡） | 64, 80 | 16–20 | router weight 400k；C80 用 K3；C64 在 K3 / K5 分界附近，要两种 K 就拆成两个 search-space 条目（每个 variant 只能一个 K）；#235 的 4+4 峰值在 C88–C104，可把 conc-list 扩到 88 / 104（K3） |
| `override_1p1d`（PP4 prefill + TP8/DP8 decode，12 卡） | 32, 48, 64, 88 | 4–11 | 保持 K5；mem 不动 |

第 4 项（stream-interval + CLIP）对三个形状都适用，加在 base 上。

K 的分界依据（#235 README "Recipe"）：InferenceX 给 K 打的分是 golden AL（K3 2.99、K5 3.61），所以 K5 的单步时间不到 K3 的 3.61 / 2.99 = 1.21 倍才划算。
单机 4+4 C64 上实测 K5 / K3 单步比：最忙 rank ≤ 3 个请求时 1.09–1.14，6–12 个时 1.13–1.19，14–18 个时 1.21–1.23。

recipe 改法示意（srtctl 的 override 按字段深合并；decode 的 `env: *environment` 要改成合并写法才能只给 decode 加变量）：

```yaml
  roles:
    decode:
      env:
        <<: *environment
        SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION: "8"
      args:
        <<: *server
        # ...现有 decode 参数...
        stream-interval: 30

override_1p1d_d4:
  frontend:
    args:
      decode-request-weight: 400000
  # ...现有 roles...

override_2p1d:
  roles:
    prefill:
      workers: 2
      args: { speculative-num-steps: 3, speculative-num-draft-tokens: 4 }
    decode:
      args: { speculative-num-steps: 3, speculative-num-draft-tokens: 4, mem-fraction-static: 0.95 }
```

## 3. 每项的依据

### 3.1 MTP K3

- stock 参数，无需补丁。InferenceX 的 `infx/srt_slurm/synthetic_acceptance.py` 从 role 的 `speculative-num-steps` 读 K，
  再从 `infx/golden_al_distribution/glm-5.2-fp8__mtp__native__default-default.yaml`（thinking_on）取 golden AL：K3 2.99、K4 3.33、K5 3.61。
- 实测：[`../../yihou/glm52.llrepro-k3-fp8-gemm-mem.packup_20261027/OPT_LEDGER.md`](../../yihou/glm52.llrepro-k3-fp8-gemm-mem.packup_20261027/OPT_LEDGER.md) 第 27 行（llr27，C96 +4.8%）；
  C64 的 K3 / K5 对比见 [`../dp-graph-variant-fix-aus-20261103/results/opt/score-k3-nohip-1node-c64.txt`](../dp-graph-variant-fix-aus-20261103/results/opt/score-k3-nohip-1node-c64.txt)、
  [`score-k5-nohip-1node-c64.txt`](../dp-graph-variant-fix-aus-20261103/results/opt/score-k5-nohip-1node-c64.txt)。
- prefill 也要带同样的 EAGLE 参数（#235 README note 10：prefill 没有 NextN 层时 KV 传输不带 draft 层 KV），所以两侧一起改。

### 3.2 decode mem 0.95

- stock 参数。ledger 第 4 行（llr2，无 custom AR、无 KF、K5）：0.95 能跑，qps +2.4% 但 ISL −6%，判"不确定"；
  第 8 行（llr8，有 custom AR + KF、K5）：HSA out of resources；第 33 行（llr33，K3 之后 0.93 → 0.95）：+1.1%；0.96 再无收益（llr36）。
- custom AR 和 graph 都占显存，#5138 关了 custom AR、不用 KF，free memory 比 Infera 栈多；仍应在每次运行时检查 `HSA_STATUS_ERROR_*`，出现就退回 0.93。

### 3.3 router decode-request-weight 400k

- Infera router 参数（`--decode-request-weight`），amd-infera 0.3.3 自带，#5138 的 `override_1p1d` 已在用；不涉及 SGLang 镜像。
- 实测：[`score-nohip-1node-full.txt`](../dp-graph-variant-fix-aus-20261103/results/opt/score-nohip-1node-full.txt)（weight 0，53,209 @ p90 61.0）对
  [`score-dpbal400k-nohip-1node.txt`](../dp-graph-variant-fix-aus-20261103/results/opt/score-dpbal400k-nohip-1node.txt)（weight 400k，52,813 @ p90 65.0，TTFT p90 9.64 → 12.39 s）。
  1,000,000 时 interactivity p90 +7.7%、吞吐持平，但 TTFT mean +22%，所以 #235 选了 400k。
- #5138 给 d4 写的理由是"KV-bound, so token-only decode demand"；#235 的 4+4 数据说明加上 400k 的请求数权重后 Pareto 仍然右上移。

### 3.4 stream-interval 30 + CLIP 8

- 都是 stock：`--stream-interval`；`SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION`（`environ.py:713`，默认 4096），decode 侧用于预分配时每个请求预留的
  `min(max_new_tokens, CLIP)`（`disaggregation/decode.py:189`、`:637`）。
- `managers/scheduler_components/output_streamer.py:588-602`：`stream_interval > 1` 时首 token 立即发出，之后每攒够 `stream_interval` 个 token 发一次
  （MTP 一步多 token 也按"未发数 ≥ interval"判断），所以 TTFT 不变。
- 实测：ledger 第 31、32 行（llr13 +0.9%，llr14 +0.35%，三次均值 42,689），在噪声内，保留是因为零风险、InferenceX 常用。

### 3.5 index_share_for_mtp_iteration: true（不建议加）

- `/perf_apps/data/models/GLM-5.2-MXFP4/config.json` 与 `zai-org/GLM-5.2-FP8` 的 `config.json` 都是 `"index_share_for_mtp_iteration": true`。
  SGLang 只在 `topk == 1` 时启用（`speculative/eagle_worker_v2.py:346-360`），后续 draft step 复用第一步的 indexer top-k。
- 规则：InferenceX `.github/codeowner-signoff-verify-prompt.md` Check 8(a) 禁止"没有模型原生支持"的 FLOP 削减，这里是 checkpoint 原生值；
  8(b) 要求 indexer sharing 有 eval 覆盖，#5138 本来就跑 GSM8K。forced acceptance 下打分不受 draft 质量影响。
- 风险：修空闲 rank metadata 的 [sglang#45156](https://github.com/sgl-project/sglang/pull/45156) 仍 OPEN；stock 的
  `model_executor/runner/eager_runner.py:434-444`（`_execute_idle`）仍对 `model_runner.attn_backend` 做 init，DP attention 下空闲 rank 可能
  memory access fault。yihou 的结论是"≥ 1.5 h soak、0 fault 才保留"。
- 实测两次，方向不一致：ledger 第 20 行（llr3，4+4 C80，+0.9%，TPOT −3.4%）；
  [`../mix-v0522-correctness-20261019/README.zh-CN.md`](../mix-v0522-correctness-20261019/README.zh-CN.md) 第 144 行（mixed，吞吐 −1.3%，
  intvty p90 略好、TTFT p90 略差），#210 据此维持 false（[`../2p1d-v0522-20261020/README.zh-CN.md`](../2p1d-v0522-20261020/README.zh-CN.md) 第 56 行）。
  两次都在噪声内，加上 MAF 风险，不值得在 #5138 上换。
  注意 `opt-aus` 的 `acc-idx0`（index_share false 的真实接受率 1.20）不能用来比较：那一批在 #45800 回移前，FP8 draft 的接受率都在 1.2 左右。

### 3.6 SGLANG_OPT_USE_TOPK_V2=false（不建议加）

- stock 开关：`arg_groups/model_hook.py:450-452` 在 HIP 上对 DSA 模型默认设 false（除非显式设置）；为 true 时 decode / verify 的 PAGED top-k 走
  V4 top-k v2 JIT kernel（`layers/attention/dsa/dsa_topk_backend.py:52-53`、`:121-131`）。
- #235 在 decode 设 false，是为了让 KF overlay 自己的 top-k 生效（ledger 第 25 行，llr7 +1.8%，标注 "KF top-k engages"）。
- 不用 KF 时 true 更好：[`../2p1d-v0522-20261020/README.zh-CN.md`](../2p1d-v0522-20261020/README.zh-CN.md) 第 55 行，两侧 true，mixed C48 同节点 +8.6%，正确性不变。
  #5138 保持 true。

## 4. 开不了的项

| 项 | #235 实测 | 为什么开不了 |
|---|---|---|
| decode custom all-reduce + `SGLANG_USE_ROCM700A=0` | +0.4%（llr6）；#235 表里加 ROCM700A=0（同时关 `DRAFT_MOE_FP8`）那一步 +2.6% | 依赖 dsa-graph-variant（[sglang#48302](https://github.com/sgl-project/sglang/pull/48302) OPEN）；`703e763d` 里没有 `dp_decode_graph_max_seq_len`。不打补丁时各 DP rank 回放不同 graph variant，乱码（#235 README note 6） |
| prefill `--pp-async-batch-depth 1` | +2.5%（llr11） | 参数 stock 有，但 stock `disaggregation/prefill.py:1660-1667` 只在 `enable_overlap` 时给 early send 记 event；PP 下 scheduler 跑在 forward stream 前面，非末块 KV 可能边写边被读，是之前乱码的主因。修复在 [sglang#45524](https://github.com/sgl-project/sglang/pull/45524)（OPEN）/ #235 的 `pp-chunk-wait` 补丁 |
| prefill `SGLANG_PP_COMM_OVERLAP=1` | +1.7%（llr21）；在 C104 的最佳栈上去掉后 −14.4% | 同上（`environ.py:782`，stock 默认 false） |
| `SGLANG_EVICT_HEADROOM_TOKENS` | +2.6–3.0%（同事数据，二手） | 环境变量本身是 #235 的 `evict-headroom` 补丁加的，stock 里没有；上游 [sglang#48047](https://github.com/sgl-project/sglang/pull/48047) OPEN |
| `SGLANG_HICACHE_BACKUP_BLOCK_QUOTA` | 未单独量化 | 环境变量是 `hicache-backup-quota` 补丁加的，stock 里没有 |
| KF kernels（sparse MLA / indexer / MXFP4 MoE） | +8.0%（llr5） | `sitecustomize.py` 运行时替换函数，InferenceX 不允许 overlay / monkey-patch；MoE kernel 没有上游 PR |
| aiter tuned GEMM CSV | +0.9%（llr32） | 挂进镜像的 aiter 安装目录，等同改镜像 |
| FP8 draft MoE（`SGLANG_GLM_NEXTN_MOE_PTPC=1`） | +1.0%（llr29） | 违反 InferenceX Check 13（draft 必须按发布精度运行；`amd/GLM-5.2-MXFP4` 的 NextN 层按 BF16 发布） |
| `MC_ENABLE_DEST_LOCAL_RAIL=1`、`MC_IB_TIMEOUT=18` | 同事 P8D8 C80：传输失败 51 → 0，+5.6%（二手） | 需要 Mooncake [#4540](https://github.com/kvcache-ai/Mooncake/pull/4540) / [#4541](https://github.com/kvcache-ai/Mooncake/pull/4541)（都 OPEN），stock Mooncake 不认这两个变量 |

sglang#48302、#45524 合入并进 nightly 后，前三项可以再评估。

## 5. 建议的验证

在 InferenceX 路径（AUS，#5138 的镜像和 Infera 0.3.3）上，一次只改一项，对照 #5138 当前配置：

1. `override_2p1d` C176：K3，再叠 decode mem 0.95。
2. `override_1p1d_d4`：C80 的 K3 vs K5；再加 `decode-request-weight: 400000`，看 C64 / C80 的吞吐和 interactivity p90；可选扩 C88 / C104。
3. base：stream-interval 30 + CLIP 8，一次计分即可（预期在噪声内，主要确认没有回退）。

每次运行都检查 decode 日志里的 `HSA_STATUS_ERROR_*`。

## 6. 依据清单

- #235 kit：`examples/sglang_pd_glm5.2_agentx/`（`env.sh`、`engine/worker.sh`、`README.md`、`patches/*.diff`），分支 `feat/sglang-pd-glm5.2-agentx`。
- #5138 recipe：`inferencex-e2e/benchmarks/multi_node/srt-slurm-recipes/glm5.2/infera-sglang/mi355x-fp4/agentx/disagg-variants.yaml`、
  `inferencex-e2e/configs/amd-master.yaml`，分支 `amd/glm52-mi355x-infera-sglang`；背景见 [`../inferencex-4629-nopatch-20261113/`](../inferencex-4629-nopatch-20261113/README.zh-CN.md)。
- SGLang 源码：`703e763d`（`python/sglang/srt/` 下的路径和行号如上文）。
- 单变量叠加的 ledger：[`../../yihou/glm52.llrepro-k3-fp8-gemm-mem.packup_20261027/OPT_LEDGER.md`](../../yihou/glm52.llrepro-k3-fp8-gemm-mem.packup_20261027/OPT_LEDGER.md)（llr1–llr36，4+4，C80 / C96）。
- AUS 计分：[`../dp-graph-variant-fix-aus-20261103/results/opt/`](../dp-graph-variant-fix-aus-20261103/results/opt/)（`score-*.txt`，3600 s，forced acceptance）。
