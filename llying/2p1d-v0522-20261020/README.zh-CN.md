# GLM-5.2-MXFP4 2P(TP4)1D(TP8)，sglang v0.5.22，138 + 140（10-20 起）

目标：在 v0.5.22 镜像上做 2P1D 的 example（`examples/sglang_2p1d_glm5.2`，分支 `feat/sglang-mix-glm5.2-tp4-v0522`），
端到端验证正确性，用融合配置测性能，再测 C176 以外的并发点，为 InferenceX 上 infera + sglang 的 Pareto 线提供数据。
每卡吞吐按 16 张 GPU 计。

## 部署

| 实例 | 节点 | GPU |
|---|---|---|
| prefill 0（TP4 / DP4 attention） | crsuse2-m2m-138（10.245.157.237） | 0–3 |
| prefill 1（TP4 / DP4 attention，PID 偏移 30000） | 138 | 4–7 |
| decode（TP8 / DP8 attention，EAGLE MTP） | crsuse2-m2m-140（10.245.159.30） | 0–7 |
| etcd / router / AgentX、GSM8K 客户端 | 138 | — |

镜像 `infera-sglang:v0522-fdbafb60`（合并 main 后的分支 `fdbafb60` 按 `Dockerfile.sglang` 构建，ID `sha256:509b8427…`，
138 构建后 `docker save | load` 到 140，两边 ID 相同）。运行配置 138 `/mnt/m2m_nobackup/liyingli_3p1d/mix-v0522/2p1d/site.env`，
运行目录 `.../2p1d/work/runs/`。

## 融合配置（example 默认值）

以 AUS 2P1D 最高吞吐 a1-spf（`../2p1d-aus-v0521-20261019/`，v0.5.21，C176 = 35,377）为底，叠加今天 mixed 的 A/B 结论
（`../mix-v0522-correctness-20261019/`）：

| 项 | 取值 | 来源 |
|---|---|---|
| prefill | mem 0.88，chunk 32768，max running 256，HiCache 250 GB/rank `write_back`（ratio 1.5、kernel、page_first），`--schedule-policy shortest-prefill-first`，`HSA_NO_SCRATCH_RECLAIM=1` | a1-spf |
| decode | mem 0.92，decode graph 到 256，max running 256，`--tokenizer-worker-num 4`，EAGLE 5 / 1 / 6 | a1-spf |
| 两侧 | `--disable-custom-all-reduce`（同时关掉 QuickReduce，覆盖 a1-spf 的 prefill `ROCM_QUICK_REDUCE_QUANTIZATION=NONE`） | mixed：custom AR 导致乱码 |
| 两侧 | `SGLANG_OPT_USE_TOPK_V2=true` | mixed：+8.6% 吞吐、正确性不变 |
| 两侧 | `index_share_for_mtp_iteration=false` | mixed：默认值无收益 |
| router | kv-aware，prefill / decode 重叠权重 20 / 2，`--session-affinity prefill`，`--pd-prefill-guard-release completion`（R1），`--decode-input-demand on`（R2） | 3P1D / a1-spf（R1、R2 现为正式参数） |
| Mooncake | 8 条 ionic，GID 1，`MOONCAKE_DISABLE_HIP_DMABUF=0`，dest-local-rail + `MC_IB_TIMEOUT=18`，dest device affinity | crsuse2 3P1D（AUS 的 DMABUF=1、NodePort 等不适用） |
| 分配策略 | early（默认） | a1-spf（prefill_complete 在 AUS 上更差） |

## 计划

1. main 合入分支（无文本冲突），lint / unit / 镜像内 unit / engine 检查。
2. 写 example（无自动门禁，手动起停），GSM8K 与 InferenceX 一致（`run_eval`），AgentX 沿用 InferenceX `918524ff`。
3. 正确性：真实 MTP 接受率起栈，smoke + GSM8K 全量 1,319 题、并发 64。
4. 性能：C176（模拟接受率 3.61），对照 AUS 35,377。
5. 并发点 C128 → C96 → C64 → C32，每点按 InferenceX 方式整栈重启（down → 等显存释放 → up → 预热 10/lane → 3,600 s）。
   每点开测前检查两台节点有无他人 GPU 进程 / 显存 / CPU 负载。

## 进度

**15:20 main 合入**（`fdbafb60`，已推送）：无文本冲突；PR #187 在 main 上的最终内容与分支已有的完全一致（合并后 `deploy/` 无差异）。
合并后检查：unit（python:3.10）2,789 passed / 149 skipped；镜像内 unit 2,579 passed / 91 skipped；engine（sglang，8 个文件）全部通过。
lint 只有 `ruff format` 失败一处：`infera/projection/core/projection/module_profilers/moe_mlp.py`，与 origin/main 完全相同，
来自 main 的 PR #205，不是本分支改动，未在本分支修改（会与 main 产生分歧）。140 上没有 `python:3.10` 且 Docker Hub 凭据过期，lint 改在 138 跑。

**15:40 第一次起栈（正确性，真实 MTP 接受率）**：prefill 1 启动失败——`port_base at 27436 is not available`。DP attention 下
sglang 从 `--port + 233` 起占 11 个端口（DP4），两个 prefill 端口只差 1（27201 / 27202）时两段重叠。改为 prefill i 用
`PREFILL_PORT + 256 i`（与 AUS harness 的 stride 256 一致），只删掉并重启 prefill 1（prefill 0 与 decode 在冷启动中，不重来），
其余照 `up.sh` 继续。KV 事件内部端口由 infera 随机挑选，两个 prefill 不冲突（17471 / 17123）。

**干扰检查（15:36）**：两节点 GPU 显存为 0；138 上他人容器 `dev_xb`、`yihou-expert8-tools-138` CPU ≈ 0；138 有一个
12 天前启动的 root python 进程（pid 2906078）持有 GPU 句柄、显存 0，之前所有性能测试期间都在，不动它。140 只有系统 `gpuagent`。
