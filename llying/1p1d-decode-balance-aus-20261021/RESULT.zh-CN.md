# 1P1D C88 decode 按请求数均衡：结果与分析

> 部署与方法见 `README.zh-CN.md`，时间线见 `operations/STATUS.zh-CN.md`。吞吐按 12 张 GPU 计（AgentX 自己算的 per_gpu）。

## 0. 摘要（持续更新）

- **第一个结果（`ka8-pmtp-w180k`：prefill 带 EAGLE 参数 + router 权重 180,000，1P1D C88，1800 s）：27,577 tok/s/GPU，interactivity
  p50 / p90 = 85.0 / 74.4**；PR 210 的 1P1D C88（crsuse2，prefill 无 MTP，router 无权重）为 27,453 @ 49.1 / 42.8。吞吐相同（prefill 是瓶颈），
  p90 交互 +74%。这一轮在测量窗口结束 2 秒后被抢占，结果由 AIPerf 的逐请求记录算出（§2）。
- 拆分两个因素需要同部署的 w0 对照（排在下一次分配）。机制上：decode 每 rank 平均只有 4.1 个在跑（PR 210 为 6.7），每步中位 41.6 ms——
  主要是 prefill 带 MTP 让 decode 草稿层拿到真实 KV、每步变快、在跑数随之减少。
- v0.5.22 C88 上最忙 rank 多 1 个请求 ≈ 多 105 万 token（v0.5.21 C176 为 17–19 万），180,000 低估了请求数；B 组请求数最忙 / 均值仍为
  1.47 / 1.82。下一组用 1,000,000。
- PR 210 的 kit 从分支本身以 `PREFILL_WORKERS=1` 在 AUS 上起栈成功（11 分钟，router 列出 1 prefill + 1 decode），只需站点环境变量；
  镜像在 AUS 上原样构建成功（约 7 分钟）。分配极短（51 / 31 / 62 分钟，两天内 9 次抢占），A/B 改为每组 1800 s。

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

## 2. 结果

| 轮次 | prefill MTP | router | 窗口 | tok/s/GPU | 请求 / rps | intvty p50 / p90 | ITL p50 / p90 (ms) | TTFT p50 / p90 / 均值 (s) |
|---|---|---|---|---|---|---|---|---|
| PR 210 1P1D C88（crsuse2，README） | 无 | 镜像（R2） | 3600 s | 27,453 | 9,970 / 2.75 | 49.1 / 42.8 | 20.4 / 23.4 | 4.94 / 9.45 / 7.29 |
| **ka8-pmtp-w180k**（AUS n02-29 + n04-33） | 有 | 新，权重 180,000 | 1800 s | **27,577** | 5,234 / 2.91 | **85.0 / 74.4** | 11.8 / 13.4 | 5.46 / 11.49 / 8.77 |

`ka8-pmtp-w180k` 的 AgentX 在测量窗口结束（00:36:13）后 2 秒被抢占，AIPerf 没写汇总；数字由 `scripts/records_summary.py` 从
`profile_export.jsonl`（随请求完成追加，5,234 个测量期请求）按 InferenceX 的定义算出（interactivity = 1 / 逐请求 full-response ITL 的
分位数；吞吐 = 窗口内完成请求的输入 + 输出 token ÷ 1800 s ÷ 12 卡），与 AIPerf 被杀前最后一次实时统计一致（TTFT p50 5.446 s、rps 2.9）。
prefill flush 核对通过（4 / 4 rank 的第一个 batch `cached=0`）。

decode（测量窗口，`decode_fit.txt` / `engine_pace.txt`）：

| | 每 rank 平均在跑 | 最忙 中位 / p90 | 请求 最忙 / 均值 中位 / p90 | 上下文 最忙 / 均值 中位 / p90 | 每步中位 |
|---|---|---|---|---|---|
| ka8-pmtp-w180k | 4.08 | 6 / 8 | 1.47 / 1.82 | 1.39 / 1.61 | 41.6 ms |
| PR 210 C88（GB300 分析文档） | 6.7 | 11 / 16 | 1.64 / 2.39 | — | ~71 ms |

步长拟合（v0.5.22，C88，593 个 3 s 分箱，R² 0.863）：`每步 ms = 20.6 + 1.70·最忙请求 + 1.63·最忙上下文(M) + 0.219·总请求 − 0.05·总上下文(M)`，
最忙 rank 1 个请求 ≈ 1,046,000 token。每步按最忙 rank 在跑数：@4 31.9、@5 35.6、@6 40.9、@7 42.1、@8 43.4、@9 49.2 ms。

## 3. PR 210 review（代码阅读 + AUS 上的验证，持续更新）

| # | 位置 | 结论 |
|---|---|---|
| 1 | `examples/sglang_2p1d_glm5.2/engine/worker.sh` | **prefill 没有 EAGLE 参数**（只在 decode 分支加）。prefill 不加载 NextN 层，PD 传输里没有草稿 KV / DSA index-K，decode 草稿层对每个 prompt 位置读未初始化显存：新进程里是 0，草稿 indexer 打分大量并列，top-k 散射 kernel 慢 15–170 倍（AUS `../2p1d-aus-v0521-20261019/RESULT.zh-CN.md` §2.13）。README 的每个参考点都是"新起部署"= 冷态，受这个影响；InferenceX 官方 AMD recipe（`amd_utils/server_sglang.sh`）明确两侧都给 MTP 参数，v0.5.13 起 Mori 后端会因状态分量数不一致直接拒绝，Mooncake 后端（本 kit）静默放过。修法已有：`../2p1d-v0522-20261020/config/example-prefill-mtp.diff`。 |
| 2 | `deploy/docker/Dockerfile.sglang` | AUS 两台节点原样构建成功（base 拉取 + Mooncake + infera + Rust router，约 7 分钟）。两台各自构建的镜像 ID 不同（非可复现构建），内容来自同一 commit 与固定版本。 |
| 3 | kit，`PREFILL_WORKERS=1`（PR 的待办项） | 从分支本身在 AUS 上起栈成功（11 分钟），router 列出 1 prefill + 1 decode。只需站点环境变量；kit 的 `ssh` 用本作业的 srun 替身即可。AgentX C88 结果见 §2。 |
| 4 | `infera/engine/sglang/args.py` `_rename_cuda_graph_max_bs` | 逻辑正确（只在新 sglang 没有旧参数、有 `-decode` 参数时改名，旧镜像不变）。小问题：用了 argparse 私有的 `_option_string_actions`；没有处理 argparse 默认允许的前缀缩写（如 `--cuda-graph-max`）。 |
| 5 | `infera/router/kv_event/responses_input.py` | 固定 `supports_inline_system = False`：对会在对话中插 system 消息、且模板支持内联 system 的模型，router 侧渲染（缓存感知哈希）会与引擎不同，只影响路由的缓存命中，不影响正确性；注释已说明。 |
| 6 | 新补丁（disagg 非末 chunk wait event、MLA absorb BMM 默认配置、KV metrics 非阻塞） | 都有锚点自检与幂等标记，头注释如实写了"是否仍必要未测"（wait event）与"v0.5.22 上未复测崩溃"（absorb BMM）。 |
