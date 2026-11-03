# 复现：decode CUDA graph variant 跨 DP rank 一致性修复的验证

本目录是自包含的：kit、E10 配方、补丁、题目、验证与判定脚本都在这里，不依赖 crusoe 上的任何路径。
背景、修法与判据的来由见 `README.zh-CN.md`。下面 `PKG` 指本目录所在位置。

## 1. 这套验证回答什么

| 问题 | 验证组 | 判据（`scripts/judge.py` 自动判） |
|---|---|---|
| 修法本身每一处行为对不对 | `unit`（镜像内单元测试） | 全部通过 |
| 第 1 部分（verify 只捕获一套 graph）是否修掉 E10 的"开头坏 / 长生成后乱码" | A（对照）× 2 与 B（修法）× 3 交替的限池门禁 | A 合计 ≥ 3 条（否则本次无检验力）；B 600 题开头坏 ≤ 1、乱码 ≤ 1；长上下文每遍捞针 ≥ 62/64、长前缀 GSM8K ≥ 60/64 |
| 第 1 部分是否修掉 E11 的乱码 | C0（E11 对照）、C（E11 + 修法）× 2 | C0 乱码 ≥ 10；C 400 题乱码 0、开头坏 ≤ 1 |
| 第 2 部分（DECODE 模式各 rank 用同一上界）机制是否成立 | N1a（不开 MTP + 修法 + 审计） | 跨 rank 不一致 0；dense 时有 > 2048 的序列 0；上界被突破 0；无上界 0；dense 仍在用 |
| 第 2 部分效果 | N0（对照）、N1（修法） | N0 乱码 > 0 时第 2 部分的必要性成立（为 0 则如实记"未观察到"）；N1 乱码 0、开头坏 ≤ 1 |
| 生产配置有没有副作用 | Bprod、Cprod（不限池，1,319 题 + 16k × 128 + 100k × 32 + InferenceX lm-eval） | 正确率 ≥ 0.96、乱码 0、开头坏 ≤ 1；16k 捞针 ≥ 62/64、长前缀 ≥ 60/64；100k 各 ≥ 15/16；lm-eval strict / flexible ≥ 0.96 |
| 性能 | scoreB、scoreC（AgentX C104，3600 s） | 只记录；预测 B ≥ 51,900、C ≥ 52,400（crusoe 同一对上的 E10：51,374，单次 ±1%） |

每次部署先证明"跑的是哪一版"：decode 容器里 6 个修复文件的 md5（`files: decode fix=6` / `orig=6`）、prefill 容器里 5 个配方文件（`prefill recipe=5`）、
环境变量、启动日志里 `DSA dual-graph enabled` 的行数（开 MTP 时：对照 = DP rank 数，修法 = 0）。这些不对，后面的数字一概不算。

## 2. 前置条件

- **两台** 8 × MI355X 节点：prefill、decode 各一台，跨机 RDMA。**同机部署不能代替**：同机时 Mooncake 走 XGMI，E10 的"开头坏"在同机上复现不出来
  （`../car-copyin-141-20261103/`）。
- 两台都能用 docker（不需要 sudo），且能 bind mount `PKG` 和模型目录；两台上路径相同。**AUS（smci355）上 NFS home 不能 bind mount**，把 `PKG` 拷到 `/perf_apps/$USER/...` 这类目录。
- prefill 节点能 `ssh` 到 decode 节点。Slurm 作业里不行时，`scripts/slurm_verify.sbatch` 自动改用 `tools/sshshim/ssh`（`srun --jobid --overlap`）。
- lm-eval（Bprod、Cprod）要在 prefill 节点克隆 InferenceX 固定提交（GitHub）并 pip 安装 lm-eval；访问不了时加 `PROD_EVAL=0`，判定里 lm-eval 一项会标 FAIL（缺失），需另跑。
- 镜像 `infera-sglang:v0522-fdbafb60`，两台都要有（第 3 步）。

## 3. 步骤

**3.1 取包**：`git clone git@github.com:AMD-AGI/Infera.git`，切到分支 `dev/pd_opt/glm_5.2_agentx`，把 `llying/dp-graph-variant-fix-20261103/` 拷到两台都能访问、docker 能 bind mount 的共享目录，记为 `PKG`。

**3.2 镜像**（在 prefill 节点，约 30–60 分钟）：

```bash
cd $PKG
INFERA_REPO=<上面的 Infera clone> SAVE_TAR=<共享目录>/infera-sglang-v0522-fdbafb60.tar bash scripts/build_image.sh
# decode 节点：
docker load -i <共享目录>/infera-sglang-v0522-fdbafb60.tar
```

`build_image.sh` 构建完会核对 11 个基线文件（`patches/BASE_MD5SUMS`）。不一致说明构建出的 sglang 与补丁的基线不同，**停下来**，不要继续（见第 6 节）。
拉基础镜像遇到 Docker Hub 凭据问题时加 `DOCKER_CONFIG=$(mktemp -d)`（匿名拉取）。

**3.3 站点配置**：`cp config/site.local.env.example config/site.local.env`，按集群填。AUS 上之前用过的值（`../1p1d-decode-balance-aus-20261021/scripts/kit_env.sh`）：
`MODEL=/perf_apps/data/models/GLM-5.2-MXFP4`、`MOONCAKE_DISABLE_HIP_DMABUF=1`（AUS 上 0 起栈即段错误）、`WORK_DIR=/scratch/$USER/vfix`（prefill 节点本地）、
`LOG_ROOT=/perf_apps/$USER/vfix-logs`。用 Slurm 作业时节点 IP / 名字不用填，由作业脚本覆盖。

**3.4 运行**（全程约 12–14 小时；可分段）：

```bash
cd $PKG
# Slurm（推荐）：2 台整机
sbatch -p <分区> -A <账户> -q <QoS> -N 2 --gres=gpu:8 --exclusive -t 14:00:00 -o <LOG_ROOT>/slurm-%j.out scripts/slurm_verify.sbatch
# 只跑一部分：在命令末尾列出步骤名，例如先看第 1 部分和第 2 部分机制
sbatch ... scripts/slurm_verify.sbatch preflight unit B1 N1a A1 B2
# 不用 Slurm、两台机器已就绪：在 prefill 节点
setsid nohup bash scripts/run_verify.sh >> <LOG_ROOT>/verify.out 2>&1 < /dev/null &
```

步骤名与顺序：`preflight unit B1 N1a A1 B2 A2 B3 C0 C1 C2 N0 N1 Bprod Cprod scoreB scoreC`（默认全跑）。作业被抢占或中断后，用剩下的步骤名重新提交即可；
`verify.out` 是追加写的，`judge.py` 按每组每次部署分别判，重复的部署都计入合计。作业结束（含取消、抢占）时 trap 会删掉两台上的 kit 容器。

**3.5 看进度**：`tail -f <LOG_ROOT>/verify.out`；每次部署的完整日志 `<LOG_ROOT>/verify/<arm>-<时间>.log`，探测原始输出 `*.probe.jsonl`、`*.longctx-*.jsonl`，
被标记的样本 `*.review.txt`；引擎日志在 `<WORK_DIR>/runs/<label>/logs/`（prefill 节点）。

**3.6 判定**：`python3 scripts/judge.py <LOG_ROOT>/verify.out`（`run_verify.sh` 结束时也会打印）。每行 `PASS` / `FAIL` / `INFO`，最后给出 FAIL 行数。
然后**逐条读每个 `*.review.txt`**：`BAD_START` 是推理以 "The" 加一串非单词字符开头、或前 60 字符里有中文的样本，`GARBLED` 是探测脚本按重复 / 标签判出的样本。
真开头坏的样子：`ThePOOLusta或少ledgearca0_cleanuposz…`（首 token 对，紧接乱码，后面常回到题目）；长生成后乱码：前面正常，几千 token 后变成 `( ( ( (` 或一串数字。
被误标的（连贯地反复验算、人名等）在结论里扣掉并写明。

## 4. 每步做什么、多久

| 步骤 | 配置 | 内容 | 约时长 |
|---|---|---|---|
| preflight | — | 两台的镜像存在；11 个基线文件 md5 与 `patches/BASE_MD5SUMS` 一致 | 5 分钟 |
| unit | — | `scripts/test_fix.py`（镜像内，挂修复文件） | 3 分钟 |
| A1 A2 / B1 B2 B3 / C1 C2 | `arm-A-gate` / `arm-B-gate` / `arm-C-gate` | 限池门禁：smoke、200 题、两遍 16k × 128 | 各 25–30 分钟，另加开 HiCache 拆栈后显存释放 10–40 分钟 |
| C0 | `arm-C0-e11` | smoke、200 题 | 20 分钟 |
| N0 / N1 / N1a | `arm-N0-nospec` / `arm-N1-nospec-fix` / `arm-N1a-nospec-audit` | 不开 MTP；200 / 400 / 400 题，N1、N1a 加 16k × 32 | 各 20–30 分钟 |
| Bprod / Cprod | `arm-B-prod` / `arm-C-prod` | smoke、1,319 题、16k × 128、100k × 32、lm-eval | 各 70–90 分钟 |
| scoreB / scoreC | 同上 | AgentX C104，3600 s 计分（`scripts/pd_perf_chain.sh`），汇总 `<LOG_ROOT>/score-vfix-*.txt` | 各约 90 分钟 |

## 5. 结果怎么读

- `files:` 一行：修法组必须 `decode fix=6`（N1a 是 `fix=5 debug=1`，审计版 runner），对照组 `orig=6`，所有组 `prefill recipe=5`。
- `start-up:` 一行：`dual_graph_lines` 开 MTP 时对照 = DP rank 数（4）、修法 = 0；不开 MTP（N 组）两者都是 4（DECODE 模式保留 dense / sparse）。
  `capture={…target_verify=…}` 修法下 verify graph 捕获时间应约为对照的一半。
- `probe:` 一行：`errors` 是请求失败（传输 / 服务端错误，返回 500），不算输出错误，但多了要看日志；`correct=a/b` 只在成功请求里算。
- `audit:` 一行（N1a）：`new_mismatch` 是修法下各 rank 选的 graph 不同的回放数，必须 0；`old_mismatch` 是同一负载下镜像原来的规则会不同的回放数（说明触发条件存在）。
- 计分：`score-vfix-B.txt` 第一行是吞吐表（与 `../best-4p4-51k-20261103/README.zh-CN.md` 第二节同列），后面是 decode 每步耗时按最忙 rank 请求数的分布。

## 6. 常见问题

- **preflight 报基线不一致**：镜像里 sglang 与补丁的基线不同，整文件挂载会覆盖掉镜像里的其他改动。办法：用 `build_image.sh` 重新按 fdbafb60 构建；
  或者把 `patches/fix.diff` 打到该镜像自己的 6 个文件上（`cd <从镜像拷出的 srt 目录副本> && patch -p1 < patches/fix.diff`），把 `config/fix-mounts.sh` 指向打完的文件，
  并在结论里写明用的是哪个镜像。`ALLOW_BASE_MISMATCH=1` 只用于排查，不能作为验证结果。
- **空闲 GPU 显存不为 0**：crusoe 新节点（`crsuse2-m2m-v3-*`）空闲时显示 10–11%。先查是谁占的（`rocm-smi --showpids`）；decode 用 0.95 的显存比例，GPU 不空会起不来或 KV 池变小。
  确认无进程、只是驱动底数后，才把 `VRAM_IDLE_PCT` 调到略高于底数。
- **起栈失败**：`verify_suite.sh` 会重试一次。见过的：Mooncake 注册某块显存 `Invalid argument`（偶发，重试即过）；RCCL 共享内存撞名（只在同机部署出现，本流程不用同机）。
- **抢占 / 超时**：用剩下的步骤名重新提交；已完成的部署结果仍在 `verify.out` 里。
- **只想验证修法、不跑对照**：不建议。A 组是判断"这次环境能不能复现问题"的依据，没有 A，B 的 0 不能说明修好了。

## 7. 文件

| 路径 | 内容 |
|---|---|
| `patches/orig/`、`patches/fix/`、`patches/debug/` | 6 个文件的镜像原版、修法、审计版 runner；`fix.diff`、`debug-runner.diff` 是对应的 unified diff；`BASE_MD5SUMS` 是 11 个基线文件 |
| `config/site.local.env.example` | 站点模板 |
| `config/recipe-e10.env`、`config/recipe-gate.env` | E10 配方、限池门禁（与 crusoe 上的原配置逐项相同，已核对） |
| `config/arm-*.env`、`config/fix-mounts.sh` | 9 个验证组；修法挂载 |
| `kit/` | 部署 kit（`2p1d-kit/llr35` 精简版 + `KIT_NO_SPEC` 开关）与 E10 的 prefill 补丁 |
| `scripts/run_verify.sh`、`scripts/verify_suite.sh`、`scripts/slurm_verify.sbatch` | 队列、单次部署检查、Slurm 双节点作业 |
| `scripts/judge.py`、`scripts/probe_stats.py`、`scripts/audit_stats.py` | 判定、探测统计、审计分析 |
| `scripts/test_fix.py`、`scripts/run_unit.sh` | 单元测试 |
| `scripts/build_image.sh` | 镜像构建与基线核对 |
| `scripts/gsm8k_probe.py`、`scripts/longctx_probe.py`、`scripts/pd_perf_chain.sh`、`scripts/score_summary.sh`、`scripts/summarize_points.py`、`scripts/analyze_pd_logs.py` | 探测、计分与汇总 |
| `tools/sshshim/ssh` | Slurm 作业内节点间 ssh 不可用时的替身 |
| `data/gsm8k_probe_prompts.jsonl.gz` | 1,319 题 GSM8K（与 crusoe 上用的题目逐条相同） |
