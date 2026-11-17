# crusoe 上跑 Infera + vLLM GLM-5.2 PD 的作业与脚本（2026-11-17 起）

本目录是工作目录 `/home/liyingli/bench_agentx/vllm-pd-20261117/` 里脚本的副本（结果与日志在工作目录的 `results/`、`logs/`、`runs/`，
各主题目录只放摘要）。脚本里的路径都指工作目录。

## 在持有作业里执行命令

spur 每个节点只允许一个进入作业的会话，而且**任何作业步骤非零退出都会结束整个作业**。做法：

1. `onnode.sh '<cmd>'`：一次短会话（节点体检）。远程命令恒退出 0，真实退出码打印为 `[rc=N]`。
2. 用唯一的会话常驻 `runner.sh`（`setsid nohup srun --jobid=<JOB> --overlap ... bash runner.sh`，与登录会话脱离）；
   它执行 `queue/<id>.sh`，写 `<id>.log` 与 `<id>.rc`，每轮更新 `queue/RUNNER.hb`。
3. `q.sh [-w] <name> '<cmd>'`：从登录节点投递命令。名字以 `L` 开头的是长任务。
4. `scripts/monitor.sh`（登录节点后台）：长任务结束、任一任务非零退出、执行器心跳超过 90 s、持有作业消失或剩余不足 30 min 时
   打印 `AGENT_LOOP_WAKE_vpd ...`，另每 30 min 一次心跳。

## 脚本

| 脚本 | 作用 |
|---|---|
| `scripts/vllm_up.sh NAME GPUS PORT [args]` | 起一个 vLLM 容器并等 `/v1/models`；`ENTRY` 可换成 Infera 包装命令。不挂宿主机 libionic；组用宿主机数字 gid（镜像里没有 render 组）；`docker run` 用 flock 串行（两个容器同时初始化同一个命名卷会失败）；`VLLM_ENGINE_READY_TIMEOUT_S=3600`（DP4 加载 GLM-5.2 超过默认 600 s） |
| `scripts/gate.sh LABEL URL METRICS_URLS` | 正确性门禁：smoke、GSM8K、16k 长上下文、真实 MTP 接受长度（vLLM 计数器前后差），一行汇总写 `logs/summary.out` |
| `scripts/metrics_snap.py` | 抓 / 比较 vLLM 计数器（接受长度 = 1 + 接受 token / 草稿次数，各 DP rank 求和） |
| `scripts/pd1_up.sh RUN` | 单节点 P4D4：etcd、prefill TP1×PP4（GPU 0-3）、decode DP4（GPU 4-7）、Infera router（与 sglang recipe 相同的 router 参数） |
| `scripts/agentx.sh RUN` | AgentX 计分（与 sglang kit 同一 InferenceX 提交 918524ff；指标前缀改为 `vllm:`） |
| `scripts/probes/` | sglang kit 的探测脚本；改动：smoke 在 `PREFILL_NUM_WORKERS=0` 时跳过 router 注册检查，GSM8K / 长上下文读 `reasoning`（v0.34 把 `reasoning_content` 改名了） |
