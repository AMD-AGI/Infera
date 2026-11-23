# GLM-5.2 P4D4（Infera + vLLM）继续优化：执行记录（2026-11-23 起）

计划见 `../vllm-pd-opt-plan-20261123/PLAN.zh-CN.md`。本目录放执行阶段的插桩、脚本、作业脚本和结果；结论、环境、问题与更正都写在这里。
若无说明，成绩都是单次运行。

## 0. 状态（11-23 10:10）

- **306119 `lyl-vpd-p0`（`once/p0-a0-b0.sh`，一个作业里先 0a 再 0b）10:02 起在 v3-186**，预计约 3.5 h。10:07 decode 日志确认插桩在 4 个
  EngineCore 进程里生效（`[vpd_steplog] patched ...`），prefill 上没有。合并成一个作业是因为本账户每人 2 个作业的名额有一个被另一个会话的作业
  （`lyl-dsv41-*`）占着。
- **第一次提交的两个作业都因我的脚本 bug 作废**：`run_once.sbatch` 里的 `R` 没有 export，作业脚本里 `$R/...` 的路径变成 `/runs/...`、`/overlay-cpupin/...`。
  306058（a0，v3-186）prefill 挂载失败、整轮无效，10:01 取消；306086（b0，v3-209）在第一步重定向就失败，没起容器，5 min 后结束。两个作业脚本已加
  `R=${R:-...}`。主线现成的 `once/p4d4-m14b-3600.sh` 也是同样的写法（没跑过），用之前要先加这一行。
- **v3-186 上的遗留**：306058 部署时 docker 按不存在的源路径在根目录下建了空目录（root 所有）：`/overlay-cpupin/vllm/v1/kv_offload/cpu/gpu_worker.py/`、
  `/runs/p4d4-a0-instr-m10cfg-3600/{instr/vpd_steplog.py,instr/vpd_steplog.pth,steplog}/`。用容器删除被 docker 授权插件拒绝（不允许挂载 `/`），
  无害，留待管理员清理。
- **计划更正（11-23 10:20）**：计划 2.2 节的 H1（新请求首步只有 1 个 token → 不均匀批次 → 4 个 rank 一起退出 FULL graph）大概率不成立。
  v0.34 调度器会给只剩 1 个 token 的新请求补 `num_spec_tokens` 个占位草稿（`v1/core/sched/scheduler.py:1358-1380`），worker 把它当 decode 跑
  （`v1/worker/gpu/model_runner.py:1426-1438`），批次仍是均匀的 1 + 3；只有同一步里另有 prefill 块（主要是抢占后的重算）时才不补齐。
  计划文档相应各处已改；阶段 1A 视 0a 结果取消。0a 的逐步日志里，补齐后的新请求计入 `n_dec` 和 `n_new_small`，只有该步不是 FULL 才归为 new 类。
  另：调度器有"DP prefill 节拍"（`prefill_schedule_interval`，MoE DP 下由 engine core 统一推迟本地 prefill 块），只能把重算的代价摊开、不减少总等待，
  对 H2 作用有限，记作备选。
- QoS：集群里没有名为 `agentx-bench` 的 QoS；本账户可用的是 `amd-agentx-qos`（每人 2 作业 / 2 节点，组 10 节点）、`amd-hyperloom-qos`、
  `amd-primus-cicd-qos`、`amd-burst-qos`（可抢占）。先用 `amd-agentx-qos`（主线 `run_once.sbatch` 的默认）。

## 0.1 a0 中途结果（11-23 11:05，测量期第 5–24 min 的 19,425 步，单次运行，整轮结束后以终版为准）

`scripts/steplog_summary.py` 在运行中的 `runs/p4d4-a0-instr-m10cfg-3600/steplog/` 上算：

| 类 | 步数占比 | 时间占比 | 步长 p50 / p90 | 相对同负载 FULL 步多出 |
|---|---|---|---|---|
| FULL | 99.3% | 90.0% | 52 / 56 ms | — |
| 有重算大块（某个 rank 上有 > 16 token 的请求，4 个 rank 一起 NONE 模式） | 0.7% | 10.0% | 693 / 1,134 ms | **9.3% 的 decode 时间（平均每步约 5.6 ms）** |

- **H1 不成立（与上午读代码的更正一致）**：新请求所在的步都是 FULL；所有非 FULL 的步都对应某个 rank 上的重算大块（145 组里 44 组的大块记录比同组早约
  1.5 s，所以第一版匹配漏掉了，归类已改为到未匹配的 rank 上找）。
- **H2 是生产与纯 decode 差距的主要来源之一**：这 19 min 里 10 次抢占，decode 本地重算约 113 万 token（每次约 11 万），按这个速率一小时约 35 次
  （m10 是 22 次）。重算块每步 0.6–1.1 s，其他 3 个 rank 跟着等。
- **稳态 FULL 步也比 sglang 慢**：最忙 rank 16 / 17 个请求时 47 / 49 ms，sglang 是 41 / 42 ms。计划 2.1 节按 A 线 64K 上下文 bench 推算的
  "纯 decode 只差 1–2 ms"低估了 AgentX 长上下文下的差距，真实差约 6–7 ms，待 trace（测量期 1500 s、2700 s）分类后定量。
- FULL 步里 `execute_model` 的 CPU 时间中位 1.8 ms。**但 py-spy（10:54，decode 各进程 20 s）显示 decode worker 93% 的采样停在同一行**：
  `ROCMAiterMLASparseMetadataBuilder.build`（`rocm_aiter_mla_sparse.py:681`，`common_attn_metadata.seq_lens[:num_reqs].cpu()`，在
  `gpu_sync_allowed()` 里），调用链是 MTP 草稿每一步的 eager 元数据重建（`AutoRegressiveSpeculator._multi_step_decode` →
  `_build_uniform_attn_metadata`；后端不支持 `update_draft_decode_metadata`，所以走不了 `_run_fused_multi_step_decode` 的单 graph 路径）。
  MTP3 每个 decode 步有 2 个草稿步，各做一次 GPU→CPU 同步：CPU 要等 GPU 上排队的工作做完才开始建元数据，这段时间 GPU 空着，异步调度的重叠也断了。
  这是计划 H3 的具体形态，GPU 空闲多少待 trace 定量。EngineCore 进程 97% 在等 worker 的结果（`shm_broadcast` 自旋），不是瓶颈。
- **快速修法 `overlay-syncskip/`**（`scripts/make_syncskip_overlay.py` 从原文件定点生成，diff 95 行）：这次同步只为把 seq_lens 截断到
  `topk_tokens`（2048）后做 aiter 分块元数据的指纹；当 `seq_lens_cpu_upper_bound` 减去余量（32）后每个真实行都不小于 2048 + query 长度、
  padding 行上界和 query 都为 0（GPU 上 padding 行由 `prepare_pos_seq_lens` 置 0）时，截断值不用同步就已知，指纹与同步路径逐字节相同。
  `VLLM_ROCM_SPARSE_MLA_SYNC_SKIP=0` 关闭。`scripts/test_syncskip.py` 在镜像里对 2 万个随机批次核对（作业里跑，通过才挂载）。b0 第 3 轮测它。
  彻底的修法是给 ROCm 后端实现 `update_draft_decode_metadata`（上游可提），视第 3 轮结果再做。
- 显存：每 rank 空闲 3.6–6.2 GiB，torch reserved 最高 274.7–276.8 GiB（总 288 GiB）。
- 据此 b0 的后两轮改为针对 H2 的部署参数（`--watermark 0.03 / 0.06`、decode 前缀缓存、两者组合），`--max-num-batched-tokens 2048` 暂缓。

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
- 三轮（11-23 11:00 按 a0 中途结果改）：r1 A/A（测具噪声、受控负载下的步分类）；r2 A 加 `--watermark 0.03`（准入时保留 3% KV 块，把抢占换成排队，
  H2；v0.34 自带的参数，`v1/core/kv_cache_manager.py:207-210, 539-569`），B 开 decode 前缀缓存（被抢占的请求重算时命中自己还没被复用的块）；
  r3 A 挂 `overlay-syncskip`（H3），B 为 sync skip + `--watermark 0.03`；A 与 r1 比、B 与 r2 的 A 比。`--watermark 0.06` 与"前缀缓存 + 水位线"顺延。
  结果在 `results/b0/`。
