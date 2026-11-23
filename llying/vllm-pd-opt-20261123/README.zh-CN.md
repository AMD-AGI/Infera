# GLM-5.2 P4D4（Infera + vLLM）继续优化：执行记录（2026-11-23 起）

计划见 `../vllm-pd-opt-plan-20261123/PLAN.zh-CN.md`。本目录放执行阶段的插桩、脚本、作业脚本和结果；结论、环境、问题与更正都写在这里。
若无说明，成绩都是单次运行。

## 0. 状态（11-23 09:50）

- **阶段 0 的两个作业已提交，在排队**（`amd-agentx-qos`，原因 `QOSGrpNodeLimit`：AgentX 组的 10 节点都在用）：
  - 306058 `lyl-vpd-a0`：插桩的 P4D4（m10 配置，3600 s），`once/p4d4-a0-instr-3600.sh`。
  - 306086 `lyl-vpd-b0`：decode-only 测具两轮（A/A、`--max-num-batched-tokens 2048`），`once/db-0b-aa-mnbt.sh`。
- QoS：集群里没有名为 `agentx-bench` 的 QoS；本账户可用的是 `amd-agentx-qos`（每人 2 作业 / 2 节点，组 10 节点）、`amd-hyperloom-qos`、
  `amd-primus-cicd-qos`、`amd-burst-qos`（可抢占）。先用 `amd-agentx-qos`（主线 `run_once.sbatch` 的默认）。

## 1. 插桩：`instr/vpd_steplog.py`

- 经 `instr/vpd_steplog.pth` 挂进 decode 容器的 `dist-packages`，每个 Python 进程都会 import，但只有设了 `VPD_STEPLOG_DIR` 才装导入钩子；
  `vllm.v1.worker.gpu.model_runner` 被导入时包住 `GPUModelRunner.execute_model` 和模块里的 `dispatch_cg_and_sync_dp`（不改 vLLM 文件）。
- 每步一行（`steps-r<dp>-<pid>.csv`）：节奏（距上一步）、`execute_model` 内的 CPU 时间、请求数 / token 数、恰好 1 + K 个 token 的请求数、少于它的
  请求数（新进 decode 的请求只算 1 个 token）、超过 16 个 token 的请求数与 token 数（抢占后的重算块）、本步新请求、被抢占数、DP 同步后的
  CUDA graph 模式与补齐后的 token 数。每 500 步记一次 `mem_get_info` 和 torch reserved / allocated。异常一律吞掉。
- 登录节点上用假模块测过：均匀 decode 记为 FULL，新请求 1 token 的步和 8000 token 的重算块记为 PIECEWISE，DP 空闲 rank 的 dummy 步也记录；
  不设环境变量时完全不生效。
- `scripts/steplog_summary.py`：按墙钟把各 rank 对齐成全局步，按 full / new（有新请求）/ big（有 > 16 token 的请求）/ other 分类，给出步数、
  时间占比、相对同负载 FULL 步的超出时间（lag 1 与 lag 2 两种归属），以及重算 token、抢占数、显存余量。合成数据上与构造值一致。

## 2. 阶段 0a：插桩的 P4D4（`once/p4d4-a0-instr-3600.sh`）

- 配置与 m10 相同（v3comm + 两边 renderer 4 + prefill GEM 0.88 + CPU 层 1.9 TB 精确锁页 `hipHostAlloc` 版，decode 不开 DP 均衡），3600 s。
- 加：decode 逐步日志；测量期 1500 s、2700 s 各一次 3 s decode trace；1200 s 时 decode 各 Python 进程 20 s py-spy（非阻塞）与线程 CPU 快照；
  开跑前查各 GPU 空闲显存（< 250 GiB 不跑）。
- 结果：标准分析在 `../vllm-p4d4-crusoe-20261117/results/a0-instr/`，阶段 0 的分析在 `results/a0/`。

## 3. 阶段 0b：decode-only 测具（`once/db-0b-aa-mnbt.sh`）

- 同一节点两套 DP4 decode（GPU 0–3 = A、4–7 = B），配置与 P4D4 decode 相同，KV 连接器换成 vLLM 自带的 `DecodeBenchConnector`
  （`startup_fill`）：新请求的 prompt KV 填假值、最后 1 个 prompt token 在 decode 上算（与 PD decode 拉完 KV 后相同），被抢占的请求整段重算。
- 负载：`data/m10-decode-arrivals.jsonl.gz`（m10 测量期 11,951 个请求到达 decode 的时刻 = 开始 + TTFT，及 ISL / OSL；3.30 req/s，
  ISL 中位 8.0 万、均值 11.7 万、最大 95 万，OSL 均值 985），`scripts/decode_replay.py` 开环回放 600–2100 s 段，同一请求同时发给 A、B；
  随机 prompt token、`ignore_eos`、模型默认采样（AgentX 不传采样参数，GLM-5.2 默认 temperature 1.0 / top_p 0.95）。
- 两轮：r1 A/A（测具噪声、受控负载下的步分类）；r2 B 加 `--max-num-batched-tokens 2048`（KV 增量与稳定性）。结果在 `results/b0/`。
