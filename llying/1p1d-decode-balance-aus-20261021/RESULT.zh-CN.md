# 1P1D C88 decode 按请求数均衡：结果与分析

> 部署与方法见 `README.zh-CN.md`，时间线见 `operations/STATUS.zh-CN.md`。吞吐按 12 张 GPU 计（AgentX 自己算的 per_gpu）。

## 0. 摘要（持续更新）

- 进行中：`k6-repro`（PR 210 kit 原样，1P1D C88）→ prefill 加 MTP → `k6-pmtp-w0` 对 `k6-pmtp-w180k`。
- PR 210 的 kit 从分支本身以 `PREFILL_WORKERS=1` 在 AUS 上起栈成功（11 分钟，router 列出 1 prefill + 1 decode），只需站点环境变量；
  镜像在 AUS 上原样构建成功（约 7 分钟）。

## 1. 为什么 R2 下 decode 仍不均衡（实验前的分析）

DP attention 下 8 个 decode rank 每步同步，所有请求在同一步里经历同一个步长，所以**交互速度的 p90 对应的是负载高峰时刻的步长**，
即最忙 rank 处在高位的时刻。AUS 2P1D C176 两轮 decode 日志（v0.5.21，R2 开，每 rank 每 40 步一行 `#running-req` / `#token`，按 3 s 分箱）
拟合（`scripts/decode_step_fit.py`）：

| 轮次 | 每步 ms = | R² | 最忙 rank 1 个请求 ≈ token |
|---|---|---|---|
| a5-pmtp C176 | 23.2 + 0.83·最忙请求 + 4.85·最忙上下文(M) + 0.169·总请求 + 0.238·总上下文(M) | 0.934 | 170,743 |
| a4-spf2 C176 | 22.9 + 0.80·最忙请求 + 4.19·最忙上下文(M) + 0.192·总请求 + 0.250·总上下文(M) | 0.941 | 192,255 |

同一窗口里每 rank 的均衡情况（a5-pmtp）：

| | 均值 | 最忙 / 均值 中位 | 最忙 / 均值 p90 |
|---|---|---|---|
| 在跑请求 / rank | 10.27（最忙中位 15、p90 19） | 1.42 | 1.67 |
| 上下文 token / rank | — | 1.08 | 1.17 |

- **R2 把 token 均衡得很好，请求数没有**：R2 的代价只有"已预约输入 token + 本请求 token"，一个 rank 可以用几个短请求填满别人一个长请求的量。
- 步长对最忙 rank 的请求数和上下文都敏感：多 1 个请求 ≈ 多 17–19 万 token。所以按"token + 权重 × 请求数"打分、权重 ≈ 18 万，
  就是让最忙 rank 的预测耗时最小的线性代价；Dynamo 的 `decode_blocks + w × active_requests` 是同一形式，GB300 的 w = 8192 块
  （请求数主导）。
- 这一组系数下，C176 若请求数完全均衡（最忙 15 → 10）约省 4 ms / 步（−6%）；C88 的相对不均衡更大（crsuse2 v0.5.22：均值 6.7、
  最忙中位 11 / p90 16），收益取决于 v0.5.22 上的系数，在复现轮日志上重拟。

## 2. PR 210 review（代码阅读 + AUS 上的验证，持续更新）

| # | 位置 | 结论 |
|---|---|---|
| 1 | `examples/sglang_2p1d_glm5.2/engine/worker.sh` | **prefill 没有 EAGLE 参数**（只在 decode 分支加）。prefill 不加载 NextN 层，PD 传输里没有草稿 KV / DSA index-K，decode 草稿层对每个 prompt 位置读未初始化显存：新进程里是 0，草稿 indexer 打分大量并列，top-k 散射 kernel 慢 15–170 倍（AUS `../2p1d-aus-v0521-20261019/RESULT.zh-CN.md` §2.13）。README 的每个参考点都是"新起部署"= 冷态，受这个影响；InferenceX 官方 AMD recipe（`amd_utils/server_sglang.sh`）明确两侧都给 MTP 参数，v0.5.13 起 Mori 后端会因状态分量数不一致直接拒绝，Mooncake 后端（本 kit）静默放过。修法已有：`../2p1d-v0522-20261020/config/example-prefill-mtp.diff`。 |
| 2 | `deploy/docker/Dockerfile.sglang` | AUS 两台节点原样构建成功（base 拉取 + Mooncake + infera + Rust router，约 7 分钟）。两台各自构建的镜像 ID 不同（非可复现构建），内容来自同一 commit 与固定版本。 |
| 3 | kit，`PREFILL_WORKERS=1`（PR 的待办项） | 从分支本身在 AUS 上起栈成功（11 分钟），router 列出 1 prefill + 1 decode。只需站点环境变量；kit 的 `ssh` 用本作业的 srun 替身即可。AgentX C88 结果见 §3。 |
| 4 | `infera/engine/sglang/args.py` `_rename_cuda_graph_max_bs` | 逻辑正确（只在新 sglang 没有旧参数、有 `-decode` 参数时改名，旧镜像不变）。小问题：用了 argparse 私有的 `_option_string_actions`；没有处理 argparse 默认允许的前缀缩写（如 `--cuda-graph-max`）。 |
| 5 | `infera/router/kv_event/responses_input.py` | 固定 `supports_inline_system = False`：对会在对话中插 system 消息、且模板支持内联 system 的模型，router 侧渲染（缓存感知哈希）会与引擎不同，只影响路由的缓存命中，不影响正确性；注释已说明。 |
| 6 | 新补丁（disagg 非末 chunk wait event、MLA absorb BMM 默认配置、KV metrics 非阻塞） | 都有锚点自检与幂等标记，头注释如实写了"是否仍必要未测"（wait event）与"v0.5.22 上未复测崩溃"（absorb BMM）。 |
