# 1P1D C88：decode DP rank 按请求数均衡（AUS，PR 210 镜像）

目标（用户 10-21 17:32）：PR [#210](https://github.com/AMD-AGI/Infera/pull/210) 的 sweep 里 2P1D 从 C176 降到 C96，interactivity p90 只从 35.7
升到 40.8；`../gb300-atom-pd-analysis-20261021/` 认为原因是 DP attention 下每步等最忙的 decode rank，而现有 router 的 R2
（`--decode-input-demand on`）按在途输入 token 均衡，不按请求数。参照 Dynamo 的 `router-decode-active-request-weight`，给 router 加
"按请求数"的项，在 1P1D C88 上 A/B，看 p90 能否提高；顺带 review / 验证 PR 210。结果与分析见 `RESULT.zh-CN.md`，时间线见
`operations/STATUS.zh-CN.md`。

## 部署：PR 210 的 kit 原样跑在 AUS 上

- 镜像：PR 210 head `aab4cc36` 的 `deploy/docker/Dockerfile.sglang`（base `lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261018`），
  两台节点各自构建（`scripts/build_image.sh`，约 7 分钟），tag `infera-sglang:pr210-aab4cc36`。
- kit：`examples/sglang_2p1d_glm5.2` 不改一个文件，站点值全部用环境变量传入（`scripts/kit_env.sh`）：`PREFILL_WORKERS=1`、
  `PREFILL0_GPUS=4,5,6,7`（与 PR 210 的 1P1D 点相同）、`SIM_ACCEPT_LEN=3.61`、`MOONCAKE_DISABLE_HIP_DMABUF=1`（AUS 的 ionic 必须）、
  `CTR=glm52-llying-2p1d-pr210`（让抢占时的容器守护能清理）。kit 里 prefill 节点 → decode 节点的 `ssh` 走本作业的 srun 替身
  （`../2p1d-aus-v0521-20261019/tools/sshshim/ssh`，节点上可再 srun）。
- 每轮（`scripts/kit_round.sh`，与 `../2p1d-aus-v0521-20261019/scripts/tune_round.sh` 同法）：停 router → prefill `/flush_cache`
  （device + host 两层，事后 `flush_check.py` 核对）→ 起本组 router → kit 的 `engine/agentx/agentx.sh`（InferenceX `918524ff`，
  10 个预热 / lane，3,600 s）→ 拷回 → 引擎节奏、decode 步长拟合、`tune_report.py` 表。

## router 改动（分支 `feat/decode-dp-request-balance`，基于 PR 210 head）

`--decode-request-weight <tokens>`（`INFERA_DECODE_REQUEST_WEIGHT`，默认 0 = 现状）：R2 选 decode 目标时，每个候选的代价从
"已预约输入 token + 本请求 token"变为再加 `weight × 该目标已持有的在途请求数`。在途请求数来自 R2 的账本（派发时 +1、decode 流结束
时 −1），与 Dynamo 的 `active_requests` 同口径；Dynamo 的打分是 `decode_blocks + decode_active_request_weight × active_requests`，
GB300 recipe 用 8192 块 / 请求（请求数主导）。新增 `infera_router_decode_demand_requests{target}` 指标；Python 启动器同名参数。
权重的取值依据：AUS 2P1D C176 两轮 decode 日志拟合（`scripts/decode_step_fit.py`），最忙 rank 多 1 个请求 ≈ 多 17–19 万 token。

实验时不重建镜像：新 router 二进制在同一 base 镜像里编译（`scripts/build_router.sh`），用 `INFERA_ROUTER_BIN` 挂进 router 容器。

## 实验设计（同一部署内 A/B）

1. `k6-repro`：kit 原样（prefill 无 MTP，router = 镜像自带）→ 复现 PR 210 的 1P1D C88（27,453 @ p90 42.8）。
2. 只重启 prefill 并加 decode 的 EAGLE 参数（GPU 0–3）：kit 的 prefill 同样没有 MTP 参数，decode 草稿层读未初始化 KV，连续几轮会
   "预热漂移"（`../2p1d-aus-v0521-20261019/RESULT.zh-CN.md` §2.13–2.14）；不修的话后一组会白捡更热的 decode。
3. `k6-pmtp-w0`（新 router，weight 0 = PR 210 行为）对 `k6-pmtp-w180k`（weight 180,000）。
