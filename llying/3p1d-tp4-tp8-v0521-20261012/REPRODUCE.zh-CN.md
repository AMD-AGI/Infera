# 复现：3P(TP4)1D(TP8) GLM-5.2-MXFP4，sglang v0.5.21，AgentX

本文说明如何从零复现目前最优的配置与结果（`RESULT.zh-CN.md` §0）。本次实验的结果是在同一个栈（r2）上逐步改动得到的
（router 重启、只重启 prefill、只重启 decode）；`scripts/reproduce_best.sh` 把这些改动合成一次完整起栈。该脚本的各个步骤都在
本次实验中单独用过，但整条流程尚未端到端跑过一次。

## 1. 环境

| 项 | 值 |
|---|---|
| 节点 | prefill-0：`crsuse2-m2m-135` GPU 0,4,5,6（GPU1 有其他租户，7 张网卡 ionic_0–6）；prefill-1 / prefill-2：`crsuse2-m2m-137` GPU 0–3 / 4–7；decode-0：`crsuse2-m2m-138` GPU 0–7；控制节点（router、etcd、AgentX 客户端）：137 |
| 拓扑 | `config/topology.3p1d.tsv`（每卡吞吐按 20 卡计） |
| 镜像 | `infera-sglang:v0521-llying-1012-mc4340-destpin`，ID `sha256:afcc2f8abff7aee8a188e574168120826f35f26fa3d3e0db3881c1e8d91e820a`。由 `image/build_image.sh` 在 yihou 的 v0.5.21 mc4340 镜像上重编 Mooncake（PR #4340 移植 + dest-local-rail + IB timeout + dest pinned HCA，补丁在 `image/`），三台节点 ID 必须一致 |
| 模型 | `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` |
| 基准 | InferenceX `918524ff`（AgentX full 模式，数据集 `semianalysisai/cc-traces-weka-062126` @23f152f6，393 条 trace，3600 s，每 lane 预热 10 个请求，seed 42） |
| 运行时目录 | 节点本地 `/mnt/m2m_nobackup/liyingli_3p1d`（缓存、客户端输出）；仓库内 `runtime/`（git 忽略） |

## 2. 配置

- `config/config.3p1d.sh`：基础配置（prefill TP4/DP4/DPA、chunk 32768、HiCache write_through；decode TP8/DP8/DPA、MTP EAGLE 5/1/6、
  模拟接受 3.61；DSA triton；grouped topk JIT 关；Mooncake 端口与网卡；AgentX 参数）。它会读取 `config/round.<ROUND>.sh` 作为本轮覆盖。
- `config/best.sh`：最优配置的覆盖 —— prefill mem 0.88，HiCache prefill-0 400 GB/rank（注册分块 512 GB）、prefill-1/2 200 GB/rank，
  decode `--tokenizer-worker-num 4`。
- router 开关（每轮由脚本传入）：`INFERA_PD_PREFILL_GUARD_RELEASE=completion INFERA_SESSION_AFFINITY=prefill
  INFERA_SESSION_AFFINITY_TTL_SECS=3600 INFERA_R2_DECODE_DEMAND=on`；客户端带 `AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true`。
- 本实验各轮的实际生效配置：`results/<variant>/c<C>/service/*-server-info.json`、`*-container.json`、`round.env`。

## 3. 步骤

```bash
cd Infera/llying/3p1d-tp4-tp8-v0521-20261012
# 0. 镜像（若节点上没有）：bash image/build_image.sh   # 构建并分发，校验三台 ID
# 1. 停掉占用这些 GPU 的旧栈（本实验为 r2）：
ROUND=r2 bash scripts/run_3p1d.sh stop
# 2. 在控制节点上后台执行完整复现（新 ROUND 名）：
ssh crsuse2-m2m-137 "cd $PWD && ROUND=best1 POINTS='160 176' setsid nohup bash scripts/reproduce_best.sh \
    > runtime/reproduce.best1.log 2>&1 < /dev/null &"
# 3. 巡检（登录节点）：
python3 scripts/tune_watch.py best1 10      # 一屏状态
python3 scripts/hang_check.py best1 180     # prefill 挂起检测
# 4. 结果：results/best-best1/c<C>/；对比表：
python3 scripts/tune_report.py best-best1/160 best-best1/176
```

`reproduce_best.sh` 的流程：安装覆盖 → `run_3p1d.sh all`（预检 → 真实接受率起栈 → census → 已知答案门禁 → decode 切到模拟接受 → 冒烟）→
KV 传输冒烟（192 × 8k token；失败则重启一次 prefill 再测）→ 后台 sampler → 30 分钟不计分预热 → 各并发点一轮 AgentX
（每轮先 flush 3 个 prefill 并校验首批 `#cached-token` 为 0）。

## 4. 预期结果（本实验，同一栈逐步调优得到）

| 并发 | tok/s/GPU | P50 / P90 interactivity | TTFT p50 |
|---|---|---|---|
| C144 | 25,269（重启后首轮，偏保守） | 47.4 / 39.6 | 2.1 s |
| C160 | 28,479 | 49.4 / 44.7 | 2.5 s |
| C176 | 30,562 | 48.9 / 44.7 | 3.1 s |

## 5. 已知问题与对策

| 问题 | 现象 | 对策 |
|---|---|---|
| 只重启 decode 后部分传输通道失效 | 某个 decode rank 收不到 KV（"Failed to get kvcache from prefill instance"），R2 会把更多请求引向该 rank | 重启后先跑 `transfer_smoke.py`；失败则重启 prefill（`restart_prefills.sh`） |
| 重启后第一轮 decode 偏慢 | 同负载下每步慢 3–17% | 先跑一轮不计分预热；用 `engine_pace.py` 核对 |
| prefill GPU 侧挂起（出现 1 次） | 某实例 4 个 scheduler 卡在 `stream.synchronize()`，GPU 100%，无日志 | `hang_check.py` 检测；py-spy 留证；`restart_prefills.sh INSTANCES=<inst>` 只重启该实例 |
| 共享 `/home` 写满 | 日志 follower 与编排脚本写失败退出 | 客户端拷回排除 `server_metrics_export.json`；每轮开跑前要求 `/home` ≥ 20 GB |
| Mooncake 偶发分片失败 | "Process failed for slice … transport retry counter exceeded"，一般被重试吸收，偶尔引起一串请求级失败 | 观察项，census 中区分窗口内与窗口结束的条目 |
