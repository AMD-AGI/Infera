# GLM-5.2 P4D4（Infera + vLLM）继续优化：执行记录（2026-11-23 起）

计划见 `../vllm-pd-opt-plan-20261123/PLAN.zh-CN.md`。本目录放执行阶段的插桩、脚本、作业脚本和结果；结论、环境、问题与更正都写在这里。
若无说明，成绩都是单次运行。

## 总览（11-24 00:00；新的在上，细节见各节）

**目前最好：a8 = 53,702 tok/s/GPU**（v3-186，3600 s，单次），**超过 sglang P4D4 的 53,351（100.7%）**，也超过不含 PTPC 的同口径目标 52.5k（102.3%）；
起点 m10 48,572（+10.6%），同节点的 a0 45,337（+18.5%）。a8 = a7 的 FP4 dispatch（步长）+ a6 的 mnbt 2048（KV 容量），两者效果叠加。
**同配置重复**：a9（v3-186）53,427，a10 A 侧（换到 v3-120）53,717；三次平均 53,615（sglang 的 100.5%），最大差 0.5%。

| 轮 | 作业 | 相对上一轮加了什么 | tok/s/GPU | 说明 |
|---|---|---|---|---|
| a0 | 306119 | m10 配置 + 逐步插桩 | 45,337 | 拆解用（有 trace / py-spy 扰动） |
| a1 | 306294 | sync skip 覆盖层（ROCm sparse MLA 元数据去掉两处主机同步） | 49,476（+9.1%） | GPU 忙碌 83–92% → 99.7%；门禁干净 |
| a2 | 306511 | decode 前缀缓存（`D_PREFIX_CACHE=1`） | 51,572（+4.2%） | 抢占后重算的超出时间 10.8% → 0.2% |
| a3 | 306707 | decode DP 引擎均衡（`INFERA_VLLM_DP_ENGINE_LB=1`） | 52,144（+1.1%） | 最忙 / 平均 1.28 → 1.13 |
| a4 | 306878 | decode BF16 GEMM 调优表（141 形状） | 作废 | prefill 在第 46 min 卡死（CPU 层 `hipMemcpyBatchAsync`，与改动无关）；卡死前同窗口 +1.4% |
| a5 | 307068 | GEMM 表 + KV 感知 DP 均衡（覆盖层，W = 14） | 52,784（+1.2%） | 步长不变；KV 满的时间少、抢占减半、TTFT p90 −1.35 s |
| **a7** | 307524 | + MoE 分发前量化成 MXFP4（FP4 dispatch 覆盖层） | **53,092（+0.6%）** | 门禁干净；hidden all-gather 约 20–29 → 5–8 µs，同负载步长 −1–2 ms，ITL −0.3 ms |
| **a6** | 307324 | a5 + decode `--max-num-batched-tokens 2048`（不含 FP4 dispatch） | **53,312（比 a5 +1.0%）** | KV 容量 +6.3%，KV 满的时间 47% → 35%，TTFT p90 −1.2 s；步长不变 |
| **a8** | 307733 | a7 + a6 的改动（FP4 dispatch + mnbt 2048 + KV 均衡 + GEMM 表） | **53,702（比 a6 +0.7%、比 a7 +1.1%）** | 超过 sglang 53,351；ITL 15.5 ms、TTFT p90 6.63 s |
| a9 | 307883 | a8 原样再跑一次（同节点重复） | 53,427 | 交互 64.6 / 60.7，ITL 15.5 / 16.5，TTFT 2.55 / 6.78；**a8 / a9 两次平均 53,565（sglang 的 100.4%），两次相差 0.5%** |

- **复现 a8**：在 `R=/home/liyingli/bench_agentx/vllm-pd-20261117` 下 `sbatch -w <node> -J lyl-vpd-a8 -t 02:45:00 --export=ALL,ONCE=p4d4-score2,SCORE_ENV=a8
  run_once.sbatch`（脚本 `once/p4d4-score2.sh`，配置 `once/score-a8.env`，都在本目录 `once/`，要拷到 `$R/once/`）。覆盖层都在作业里从镜像原文件生成并测试：
  `scripts/make_syncskip_overlay.py`、`scripts/make_dplb_kv_overlay.py`、`scripts/make_fp4dispatch_overlay.py`；GEMM 表 `gemm-tune/glm52_decode_bf16_tuned_gemm.csv`。
  decode 相对 m10 的变化：`D_PREFIX_CACHE=1`、`--max-num-batched-tokens 2048`、环境 `INFERA_VLLM_DP_ENGINE_LB=1 VPD_DP_LB_KV_WEIGHT=14
  VPD_DP_LB_KV_CAP_TOKENS=3141120 VPD_FP4_DISPATCH=1` + 四份覆盖层（sparse MLA、core_client、naive_dp_ep、rocm_aiter_moe）+ GEMM 表；prefill 不变。
  改计算的覆盖层（sync skip、FP4 dispatch）都过了门禁（a1、a7）。
- **剩下的差距**（a5 对 sglang）：decode 每 token 慢（ITL 15.8 对 14.3 ms），TTFT 反而快（2.61 对 4.20 s）。步长里 DP 不均约 1.9 ms / 步
  （只靠到达时选 rank 的下限）；MoE 已在带宽上限；dense GEMM 调表后约 4.9 TB/s；hidden all-gather 是 a7 针对的部分。
- **试过 / 判断过不做的**：H1（新请求首步不均匀）不成立；MTP 的 BF16 MoE 已在带宽上限（调优空间 0.1–0.2 ms）、草稿量化受 InferenceX Check 13 限制；
  把两次 4 字节小 all-gather 打包（拼接 / 拆分的小 kernel 抵消收益）；lm_head 在带宽上限。
- **上游**：sync skip 第一部分 = vLLM #58710（open，留言稿 `../upstream-vllm-20261117/COMMENT-vllm-58710.md`，未发），第二部分 main 上已由 #66400 删除；
  KV 感知 DP 均衡、FP4 dispatch（naive 分发）是新的上游候选，待 a7 结果后写进 `../upstream-vllm-20261117/CANDIDATES.zh-CN.md`。都没有对外发。
- **事故 / 环境**：a4 的 prefill 卡死（同主线 s2c 一类）；v3-186 dmesg 一直有 ionic `IONIC_CMD_LIF_SETPHC` 超时（a3 之前就有，与卡死无关）；
  `tune_decode_gemm.sh` 的超时不生效（容器 PID 1 是 bash），已改 `--init`。

## a18 B 侧（Triton top-k 第三版：第二版 + 每行升序）= 54,484 tok/s/GPU（11-24 15:30，v3-231，单次；同节点 A 侧 = vLLM 自带 top-k 进行中）

| | a18 B 侧 | a15 B 侧（vLLM 自带 top-k） | a8 配置三次平均 | sglang P4D4 |
|---|---|---|---|---|
| tok/s/GPU | **54,484** | 53,897 | 53,615 | 53,351 |
| 稳态 300 s 起 | 57,254 | 56,642 | 约 56,300 | — |
| 交互 p50 / p90 | 69.5 / **63.6** | 65.6 / 60.9 | 约 64.7 / 60.5 | 69.9 / 60.8 |
| ITL p50 / p90（ms） | **14.4 / 15.7** | 15.2 / 16.4 | 约 15.5 / 16.6 | 14.3 / 16.4 |
| TTFT p50 / p90 / 均值（s） | 2.53 / 7.23 / 4.13 | 2.42 / 6.70 / 3.79 | 约 2.5 / 6.7 / 3.8 | 4.20 / 9.23 / 5.65 |

- trace（dp1）：top-k 四个 kernel 约 17 + 17.5 + 27.9（select）+ 9.4（`_sort_rows`）≈ 72 µs / 次（aiter 105–186）；**稀疏 MLA 31.0 µs**
  （aiter 有序输出时 26–29、第二版乱序 36–40），排序基本收回了 MLA 的局部性。
- 比 a8 配置平均 +1.6%、比 sglang +2.1%；交互 p50 与 sglang 持平、p90 超过。同节点成对的差值等 A 侧（约 17:30）。
- **未解决：门禁真实接受 3.407**（第四次偏低约 1.2%）。计分用模拟接受不受影响，但上线前必须弄清楚，暂不进推荐配置。

## 复盘（11-24 14:10）：为什么最近几轮提升有限、接下来怎么做

**一句话**：去掉浪费的阶段（同步空等、重算）已经结束，GPU 已经 99% 忙碌；剩下的 decode kernel 项在吞吐上天然只有约 1%。还没到硬瓶颈，
但空间在 TTFT / 准入这类结构性改动上，不在继续抠 kernel。用户 14:05 选定下一步主攻 **TTFT 与 decode 准入**。

1. **周期拆解（a6 / a8）**：tok/s/GPU 主要是输入 token（ISL 约 12 万），吞吐 ∝ 完成请求数 = 104 条通道 / 通道周期。周期约 29.5 s =
   客户端思考 / 工具间隔约 8.3 s（28%，服务端改不了）+ TTFT 约 3.8 s（13%）+ decode 约 17.3 s（59%，约 1,000 token × ITL）。
2. **step 时间对吞吐的弹性只有 0.3–0.6**：理论上 step 快 1% 周期只短 0.6%；实际上 decode 变快后请求更早回到 prefill、prefill 排队变长，又抵掉一部分。
   实测：a3 ITL −3.7% → 吞吐 +1.1%；a7 ITL −1.9% → +0.6%；**a16（Triton top-k）ITL 15.2 → 14.3 ms（追平 sglang）、交互 p50 70.0（超过 sglang 69.9），
   吞吐却只有 52,924**——这台节点 prefill 慢（prefill 引擎排队 2.31 s、计算 1.11 s，其他几轮 1.13 / 0.82 s），TTFT 均值 6.27 s。
   a1 的 +9.1% 和 a2 的 +4.2% 不受这个弹性限制，因为回收的是整块空等 / 重算时间。
3. **剩下的 decode 项都小**：a8 trace 每步 48 ms = MoE 17（约 6 TB/s，到顶）+ dense 11（接近现有 kernel 上限，a12 加 hipBLASLt 也只 −0.1 ms）+ 通信 6.7
   （含等最慢 rank 约 1.5）+ top-k 3.2 + 稀疏 MLA 2.9 + indexer 1.7 + 其他。同负载 step 与 sglang 的差最多约 9% → 吞吐最多 3–5%，分摊在好几项上。
4. **噪声与迭代成本**：单次波动约 0.5%，和单项收益同量级，每项都要同节点成对、重复；一轮 P4D4 2–4.5 h。
5. **我做法上的问题**：
   - Triton top-k 第一版只在不相关的合成分布上测过就上 P4D4，连续两轮（a11、a14）线上失败，约 4 h 节点时间；乱序输出拖慢稀疏 MLA 也是上线后才发现。
     **改进**：上 P4D4 前用能复现线上行为的输入验证（已有位置相关的合成分布，下一步尽量抓真实 logits / 真实负载片段），并看下游 kernel（MLA）的时间。
   - 1% 以下的项（GEMM 重调、vLLM 自带 top-k）事前没按弹性估上限。**改进**：每项先用"step 变化 × 0.3–0.6"或"TTFT 每秒约 3%"估吞吐上限，低于 1% 的不单独占 P4D4。
   - 环境问题（v3-186 盘满被 drain、v3-176 残留进程、启动失败被挂起、a12 被外部取消）占了不少时间；**改进**：成对作业不钉节点、排除已知坏节点，两个名额并行。
6. **Triton top-k 的现状**（14:45 补充：-inf 屏蔽的推测排除——`candidate_blocks` 只用于 DeepSeek V4.1 的两级选择，GLM-5.2 不走；
   门禁里 GSM8K 一直正常、只有真实接受低，更像只影响 MTP 草稿层的 indexer / top-k 调用，待查）：线上 top-k 从 105–186 µs 降到约 62 µs（第二版）/ 约 70 µs（第三版，排序）；但门禁真实接受长度四次都低约 1.2%
   （3.41 / 3.427 / 3.416 / 3.407，排序后也没恢复；不开时 3.455–3.477），原因未明（怀疑 -inf 屏蔽位置的处理与 aiter 不同，待查）。在弄清之前不进推荐配置。
7. **下一步（TTFT 与 decode 准入）**：a8 的 TTFT ≈ prefill 引擎 2.23 s（排队 1.13 + 计算 0.82）+ decode 引擎 1.25 s（排队 0.82，含 KV 拉取）。
   按 a5 / a6 的趋势，TTFT 每降 1 s 约 +3% 吞吐。候选：prefill 侧按未命中 token 的短作业优先（sglang 上 SPF 比 FCFS +15%）、prefill 计算效率、
   decode 侧 KV 拉取与准入的排队。先拆清楚每段的分布，再定实验。

## TTFT 拆解（11-24 14:30，a8 与 a15 B 侧一致；`scripts/ttft_analyze.py`、prefill / decode 日志的 10 s 计数 + Little 定律）

| 段 | 均值 | 依据 |
|---|---|---|
| 客户端 TTFT | 3.83 s（p50 2.53、p90 6.63） | AIPerf |
| prefill 段（router 测） | 2.37 s（62%） | `prefill_usage_matched.jsonl` |
| ├ 排队：等 CPU 层读回 KV（Deferred） | 约 0.53 s | prefill 平均 Deferred 1.89 个 / 3.55 req/s |
| ├ 排队：等准入（预算 / 队头） | 约 0.47 s | Waiting 3.57 − Deferred 1.89 |
| └ 运行（计算 + PP4 流水线） | 约 0.72 s | Running 2.54 个 |
| decode 段及其余 | 约 1.45 s | TTFT − prefill 段 |
| ├ 排队：KV 拉取中 | 约 0.50 s | decode 4 个引擎合计 KV fetch in progress 1.79 个 |
| └ 排队：等开始拉取（多半等 KV 空间） | 约 0.35 s | KV fetch waiting to start 1.26 个 |

- **小请求在排队上吃亏最多**：未命中 < 2,048 token 的请求占 53%，prefill 段均值 2.06 s，按计算速度只需约 0.1 s。prefill GPU KV 只用 12.8%，
  不是 KV 限的；计算出的 prompt 吞吐 16.7k tok/s，忙一半左右（PP4 下单个请求只能用到流水线的 1/4，所以单请求速度不代表容量）。
- 候选：(1) prefill `--long-prefill-token-threshold`（长请求每步最多占这么多 token，剩下的预算给新请求，针对 0.47 s 的准入排队）；
  (2) CPU 层读回（0.53 s）：批量拷贝描述符上限（现 512）、GPU 本地前缀命中；(3) decode 的拉取与等 KV 空间（0.85 s）。
- **CPU 层读回本身不慢**（a8 测量期 OffloadingConnector 指标汇总）：8,256 次读回共 17.2 TB、合计 586 s，每次约 2.1 GB / 71 ms（约 29 GB/s）；
  但请求在"KV fetch in progress"里平均约 0.4 s——读回在某一步发起，完成要到之后的某一步才被调度器发现，延迟按 prefill 的步数计。PP4 下一个 8,192 token
  的分块依次走过 4 级（忙时约 3.3 万 tok/s → 每级约 0.25 s），这也是"运行 0.72 s"的来源。所以 prefill 的分块越大，读回的发现延迟和小请求的流水线延迟都越长。
  下一个实验：prefill `--max-num-batched-tokens 4096`（a20，接在 a19 后、同节点只跑 B 侧）。Mooncake（prefill→decode）每次平均 200 ms、1.33 GB。
- **a19（309209，v3-037，`once/p4d4-pair3.sh`）**：B = a8 配置 + prefill `--long-prefill-token-threshold 4096`，A = 同节点 a8 基线。`p4d4-pair3.sh` = pair2b
  + 每侧的 prefill 参数 / 环境（`S_P_ARGS` / `S_P_ENV`）。a16 的 A 侧取消（它评价的第二版已被 a18 的第三版取代）。

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

## 0.0 a0 终版结论（11-23 11:50；306119，v3-186，m10 配置 + 插桩，3600 s，单次）

- **成绩 45,337 tok/s/GPU**（3.073 req/s，11,152 / 8，ITL 16.9 / 20.1 ms，TTFT 2.99 / 13.14 / 6.56 s）：比 m10（48,572，v3-006）低 6.7%，在节点间
  7–10% 的波动内，另有两次 trace 与 py-spy 的扰动；本轮只用于拆解，不作成绩比较。结果在 `results/a0/`（标准分析在
  `../vllm-p4d4-crusoe-20261117/results/a0-instr/`）。
- **decode 每步的两项额外开销（相对 sglang 的同负载步长 41–42 ms）都已定位，合计约占 decode 时间的 20%**：

  | 项 | 证据 | 大小 |
  |---|---|---|
  | H2：抢占后在 decode 本地整段重算 | `steplog_summary.txt`：58,748 步里 0.8% 带重算大块，每步 p50 751 ms，4 个 rank 一起 NONE 模式；20 次抢占，重算约 367 万 token | 相对同负载 FULL 步多出 **10.8%** 的 decode 时间 |
  | H3：元数据构建里的 GPU→CPU 同步让 GPU 空等 | `trace-gaps.txt`：8 份 trace（4 rank × 2 次）GPU 忙碌 83–92%，每步空闲 4–8 ms（sglang 0.2–0.8 ms）；每步 4 次阻塞的 `hipMemcpyWithStream`（CPU 在里面共等约 34 ms）对应目标 1 次 + 草稿 3 次元数据构建里的 `seq_lens.cpu()`；另有约 2 次 `hipStreamSynchronize`（指纹变化后重算 aiter 分块元数据之后） | FULL 步 47 / 49 ms（最忙 rank 16 / 17）里约 **6 ms**；GPU 实际忙碌约 41–43 ms，与 sglang 同负载步长相当 |

- **H3 的机制**：空闲分两块。(a) 每步一次 1–2.5 ms 的空隙出现在目标模型第 0 层中间：CPU 刚做完阻塞拷贝（GPU 已排空）就发起 `hipGraphLaunch`，
  而 ROCm 的这次调用在 CPU 上要 10 ms、逐个提交节点，GPU 跑得比提交快，于是在图中间等提交；(b) 草稿元数据 eager 构建的一串小 kernel
  （H2D 拷贝、elementwise、`generate_sparse_seqlen`、rocprim cumsum、`get_mla_metadata_v1`）之间的空隙，每步 2–4 ms。两块都是因为 CPU 被同步卡住、
  跑不到 GPU 前面。去掉同步后，CPU 每步要做的事（约 12 ms 的 graph 提交 + 几 ms 的元数据构建）远少于 GPU 的 40 多 ms，应能并行掉大部分。
- **结论修正**：计划 2.1 节说"纯 decode kernel 与 sglang 只差 1–2 ms、生产多出约 6 ms 来自 PD 调度"——实际是"GPU 忙碌时间与 sglang 相当，
  多出的约 6 ms 是 host 同步造成的 GPU 空闲（H3），再加上约 11% 的抢占重算（H2）"。H1（新请求首步）不成立。
- 采样器：生产里没有 `_topk_topp_kernel`（模拟接受走另一条采样路径），2.3 节"采样 0.57 ms / 步"只在 A 线的真实采样下存在，不影响计分。

## 0.00005 下一步：indexer 的 decode top-k（11-24 06:00）

- **用户 05:50 的要求**：对齐 sglang 只是初步目标，继续按瓶颈推高 tok/s/GPU、降低 ITL。
- **a8 trace 拆解（dp1，67 个真实步，每步 48.0 ms）**：MoE GEMM 16.95（MXFP4 两段 15.1 + MTP 的 BF16 MoE 1.8）、dense 11.0（o_proj 两种 kernel
  约 3.5、lm_head × 4 约 1.2）、通信 6.7（含等待；hidden AG 3.6、RS 2.2、4 字节 AG 0.9）、**indexer top-k 3.16**、sparse MLA 2.9、indexer 打分 1.74、
  norm 1.33。**top-k 是 aiter 的 `radix_topk_one_block_decode`（每行一个 block），每次 143.6 µs、每步 22 次**；a1 时是 66 µs。
  a8 逐步日志里 rank 0 约一半的步上有 ≥ 500K 的请求（集中在 500K–800K），一个 block 一行的 kernel 时长取决于最长的行。
- 候选：A 线移植的 sglang #47872 单遍 top-k（`../vllm-decode-trackA-20261118/scripts/kf/vllm_triton_topk_decode.py`：每行估阈值，然后在固定的
  常驻网格上按 8K 一块扫一遍所有行——长行被拆给多个 block，再精确选出 top-k）。A 线在均匀 40K–200K 上只快 1.33 倍；生产的长尾下应更明显。
  先在生产形状上测（`scripts/bench_topk_prod.py`，`once/bench-topk-prod.sh`），有收益再用覆盖层接入（`scripts/make_tritontopk_overlay.py`，
  `VPD_TRITON_TOPK=1`，只在 fp32、≤ 128 行、每行有长度时启用，否则走原来的选择）+ 门禁 + 计分。
- **生产形状的 top-k 基准（308419，v3-006，`results/topk-prod/`，CUDA graph 回放计时，全部精确、0 个错行）**：

  | 场景（每请求 4 行） | 行数 | 最长 / 平均上下文 | aiter（现用）µs | vLLM 自带 µs | Triton #47872 µs |
  |---|---|---|---|---|---|
  | 16 请求，对数正态 | 64 | 382K / 79K | 83.8 | 83.1 | **33.4** |
  | 16 请求 + 一个 700K | 64 | 700K / 137K | 147.4 | 105.8 | **38.9** |
  | 20 请求 + 600K、800K | 80 | 800K / 216K | 167.5 | 109.0 | **46.4** |
  | 12 请求，都 ≤ 300K | 48 | 300K / 163K | 70.1 | 68.9 | **34.9** |
  | 24 请求 + 一个 750K | 96 | 867K / 193K | 181.3 | 122.5 | **47.3** |

  "一个 700K"一行的 aiter 147 µs 与生产 trace 的 143.6 µs 吻合；Triton 39 µs，**每次约省 108 µs、每步 22 次约 2.4 ms（约 5% 的步长）**。relu 分布结果相同。
- **a10（308422，v3-120，`once/p4d4-pair.sh`，成对 A/B 同节点）**：B = a8 配置 + Triton top-k（先门禁，再计分），A = a8 配置（这台节点上的基线）。
  **B 侧没部署起来**（06:11）：docker 报 `Duplicate mount point: .../vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`——P4D4 的 decode（`D_MODE=v3comm`）
  本来就挂了这个文件的覆盖层（U12 gluon paged logits，`$R/overlay-v3comm/`），我的补丁是按镜像原文件做的。覆盖层测试都过了（`results/a10/overlay-build-test.txt`），
  门禁判定为不干净只因部署失败。A 侧（a8 配置在 v3-120 上的基线）照常在跑。
- **a10 A 侧（a8 配置在 v3-120 上）= 53,717 tok/s/GPU**（07:46，`results/a10/A/`）：交互 64.8 / 60.8，ITL 15.4 / 16.4 ms，TTFT 2.49 / 6.56 / 3.78 s，
  稳态 56,410。与 v3-186 的 a8 53,702、a9 53,427 一致——**a8 配置三次（两台节点）平均 53,615，最大差 0.5%**。这台节点上 aiter top-k 每次 105 µs（每步 22 次，约 2.3 ms）。
- **修正（`once/p4d4-pair2.sh`）**：Triton top-k 补丁改在 v3comm 那份文件上做（补丁同样能打上，上下文一致），这一侧用 `D_MODE=stock`，自己挂 v3comm 的
  三份文件（ops 换成补过的）并带上 `VLLM_ROCM_DSA_GLUON_PAGED_MQA_LOGITS=1`，与 v3comm 等价；条件再加 `compress_ratio == 1`。
  **a11（308473，钉 v3-120，`PAIR_SIDES=B`）** = 只跑 B 侧（门禁 + 计分），与 a10 的 A 侧同节点比较。
- **a11 门禁干净**（08:03，`results/a11/B/gate.txt`）：smoke 6/6，GSM8K 198/200，乱码 0，16k 32/32、长前缀 31/32；**真实接受 3.41**，比之前几次门禁
  （3.455–3.477）低约 1.3%。Triton 与 aiter 都是精确 top-k，只在同分值时选哪个 key、以及输出顺序（注意力求和次序）上不同，可能让个别生成分叉；
  计分用模拟接受，不受影响。待计分后在同节点把基线和 Triton 的门禁各跑两次，量接受长度本身的波动再下结论。08:04 起计分。
- **a11 线上失败，09:00 取消**：测量期第 5–25 min decode 生成 1,972 tok/s（a10 A 侧 4,182），decode 排队 35.7；FULL 步平均 78.8 ms（基线约 47），
  长请求在场时单步 170–195 ms。trace（dp1）：`_topk_est` 9 µs、`_topk_collect` 19 µs 正常，**`_topk_select` p50 3,990 µs、p75 7,145 µs**——走进了
  "候选不足或溢出时整行精确选择"的慢路径。推断：真实 indexer logits（各头 ReLU 点积的加权和）在长上下文里有大量恰好为 0 的值，正值不到 2,048 个时
  第 k 大就是 0、同分的有几十万个，候选缓冲（8,192）溢出。我的基准里 relu 分布加了 `0.01 × x` 项，没有真正的 0，所以没测出来。
  a11 / a13 已取消；Triton top-k 需要先修同分值的处理（等于阈值的只取够数）再试。
- **Triton top-k 的同分值修复（09:40，`scripts/vpd_triton_topk_decode.py`，A 线移植的副本）**：est 在采样上再找出精确的第 t 大 key（kt），若按比例
  它在整行里会出现超过 2,048 次（`_TIE_MIN`），就只收集 `> kt` 的（候选不会溢出），select 把这些全部选上，缺的 k − n 个按行序取 `== kt` 的元素
  （通常第一个 4K 块就够，同分元素可以互换，仍是精确 top-k）；不够时才退回整行精确路径。`test_tritontopk.py` 加了长行几乎全 0 的用例，
  `bench_topk_prod.py` 加了 `zsparse`（0.15% 为正，700K 行的第 k 大是一个被 70 万个位置共享的 0，即 a11 的形状）、`zmid`、`mixed` 分布，
  同时测 A 线原版和修复版（`once/bench-topk-prod.sh` → `results/topk-prod2/`，308846，等空闲名额）。pair2 / pair2b 改挂修复版。
- **同分值分布下的 top-k 基准（308846，v3-140，`results/topk-prod2/`，全部精确、0 个错行，µs）**：

  | 分布 | aiter（现用） | vLLM 自带 | A 线原版 | **修复版** |
  |---|---|---|---|---|
  | zsparse（0.15% 为正，长行第 k 大是被几十万位置共享的 0） | 612–1,761 | 303–981 | 3,300–10,114 | **23–37** |
  | zmid（1% 为正；短行里正值不到 k） | 276–412 | 303–980 | 2,387–4,892 | **35–49** |
  | mixed（一半是 0） | 158–434 | 87–203 | 33–48 | 38–52 |
  | randn | 70–180 | 69–122 | 33–47 | 38–52 |

  A 线原版在同分值分布下复现了 a11 的慢路径（4–10 ms）；修复版在各分布下都在 23–52 µs，比原版多约 5 µs（est 多 15 步搜索）。现用的 aiter
  遇到大量同分值也会到 0.3–1.8 ms。**a14（308884，不钉节点，`p4d4-pair2.sh`）**：B = a8 配置 + 修复版 top-k（门禁 + 计分），A = 同节点 a8 基线。
- **a14 换了三次节点才起来**：308884 分到 v3-176，GPU 0–3 只剩 100 GiB、4–7 只剩 14–16 GiB、共享内存 1,108 GB（之前另一个会话的作业留下的进程），
  作业按空闲显存检查直接退出；308891 反复启动失败（"node has unfinished job cgroup cleanup"），重排 5 次后被挂起（`JobHoldMaxRequeue`）；
  **308972 钉到空闲的 v3-239，10:32 起**，单元检查通过（含长行几乎全 0 的用例），10:36 开始门禁。
  **门禁干净**（10:49）：GSM8K 198/200，乱码 0，16k 32/32、长前缀 32/32，真实接受 3.427。Triton top-k 两次门禁（3.41、3.427）都比不开时的四次
  （3.455–3.477）低约 1%，可能是真的：同分值时修复版按行序取最前面的 key（上下文开头），aiter 取哪些未知（若偏向最近的 token 就会有差别）。
  计分用模拟接受，不受影响；之后有空闲 GPU 时看 aiter 在同分行上取的位置，必要时改成从行尾取再比门禁。10:50 起计分。
- **a14 修复版上线仍然慢，11:47 取消**：测量期第 5–25 min decode 生成 2,002 tok/s（基线 4,182），FULL 步平均 78 ms；trace 里 `_topk_est` 15 µs
  （修复版生效）、`_topk_collect` 19 µs，**`_topk_select` p50 3,982 µs、p90 7,513 µs**——仍有一半以上的调用进了整行慢路径，同分值不是（唯一的）原因。
  推断：一次调用约 88 行并行、时长取决于最慢的一行，单行失败率只要 1–2% 就让多数调用变慢；线上 logits 在位置上强相关（相邻 token 相关度相近、
  热点集中），64 段 × 128 连续元素的采样有效样本远少于 8,192，阈值估计时高时低，候选不足或溢出。基准里的合成分布没有这种相关性。
  **下一步两条**：(1) 先上低风险的 vLLM 自带 top-k（`VPD_TOPK_IMPL=native`，精确、确定性，基准里长行比 aiter 快：randn 69–122 对 70–180 µs，
  同分值分布 303–981 对 612–1,761 µs）——**a15（309044，v3-239，`p4d4-pair2.sh`）**：B = a8 + native top-k（门禁 + 计分），A = 同节点基线；
  (2) Triton 版做稳：更多更短的采样段、放宽 OVER / CAND，慢路径换成 8 位基数选择，并用带位置相关性的合成数据测单行失败率。
- **Triton top-k 第二版（`scripts/vpd_triton_topk_decode.py`）**：CPU 仿真（`scripts/sim_topk_est.py`，700K 行，候选数落在 [k, CAND] 外即走慢路径）：

  | 规则（采样段 × 段长，OVER，CAND） | iid | AR(1) ρ 0.995 | 分段 + ReLU | 热点区 |
  |---|---|---|---|---|
  | 原版 64 × 128，2，8,192 | 0 | 55.8% | 45.0% | 50.0% |
  | 512 × 16，2，8,192 | 0 | 13.3% | 0.8% | 0 |
  | **2048 × 4，4，16,384**（1,000 行） | 0 | 0（候选 4,480–11,776） | 0 | 0 |

  据此改成 2048 段 × 4、OVER 4、CAND 16,384（select 用 16 个 warp），整行慢路径的阈值搜索从 31 遍按位搜索改成 4 遍 8 位基数直方图（一次失败的代价从
  约 4 ms 降到约 0.1–0.2 ms，估算）。基准加了位置相关的 `ar`、`segrelu` 分布（`once/bench-topk-prod3.sh` → `results/topk-prod3/`，309050）。
- **第二版基准（309065，`results/topk-prod4/`，全部精确、0 个错行，µs）**——位置相关的 `ar` / `segrelu` 复现了线上的失败（A 线原版 4,140–7,589 µs，
  `results/topk-prod3/`），第二版正常：

  | 分布 | aiter（现用） | vLLM 自带 | **第二版** |
  |---|---|---|---|
  | ar（AR(1) ρ 0.995） | 66–179 | 65–114 | **53–64** |
  | segrelu（2K 段 + ReLU） | 83–225 | 125–196 | **55–64** |
  | zsparse | 613–1,770 | 303–988 | **26–39** |
  | randn | 70–182 | 70–123 | **54–69** |

  长尾场景下比 aiter 快 2.5–3.5 倍；比第一版在 randn 上慢约 15–20 µs（CAND 加倍、select 更宽），换来稳健。第一次编译失败是 `tl.static_range` 里
  `sh: tl.constexpr` 被重复赋值（已改成直接写表达式），以及 while 循环 / 分支里用 Python 整数做初值（改成由 int32 标量派生）。
  **a16（309067，v3-113，`p4d4-pair2.sh`）**：B = a8 + 第二版 Triton top-k（门禁 + 计分），A = 同节点基线；与 a15（vLLM 自带 top-k，v3-239）并行。
- **a15 B 侧（vLLM 自带 top-k，v3-239）= 53,897 tok/s/GPU**（13:31，`results/a15/B/`，单次）：交互 65.6 / 60.9，ITL 15.2 / 16.4 ms，TTFT 2.42 / 6.70 / 3.79 s，
  稳态 56,642，门禁接受 3.456（正常）。比 a8 配置三次平均 53,615 高 0.5%（在波动边缘）；测量期第 5–25 min 生成 4,201 对基线 4,182。
  A 侧取消（native 收益小，把名额让给 Triton 的验证）。
- **a16（Triton 第二版，v3-113）线上**：trace 里 top-k 三个 kernel 17.2 / 17.6 / 27.8 µs（select p90 28.1，不再走慢路径），每次约 62 µs（aiter 105–186）；
  **但稀疏 MLA kernel 从 26–29 µs 变成 39.5 µs**（a14 也是 35.9），每步多约 1 ms，吃掉了 top-k 的大半收益——Triton 按候选到达顺序输出索引，
  MLA 读 KV 的局部性变差。aiter / vLLM 自带按扫描顺序输出（位置升序）。
- **第三版 = 第二版 + 每行升序排序**（`_sort_rows`：每行一个 program，`tl.sort` 2,048 个，-1 排最后；默认开，`VPD_TOPK_SORT=0` 关）：
  基准（309175，`results/topk-prod5/`）精确、0 错行，比第二版多约 8 µs（ar 61–73、randn 63–77、zsparse 34–48 µs）。
  **a18（309187，v3-231，`p4d4-pair2.sh`）**：B = Triton 第三版（门禁 + 计分），A = vLLM 自带 top-k（当前最好的配置）同节点。a17 取消（a18 的门禁同时回答接受长度）。
- **门禁里的真实接受长度**：Triton top-k 三次都偏低——a11 3.41、a14 3.427、a16（第二版）3.416；其余五次 3.455–3.477，含 a15（vLLM 自带 top-k，
  同样 `D_MODE=stock` 挂载）3.456，所以不是挂载方式，是 Triton top-k 本身，约 −1.2%。选出的集合是精确的、同分值也都偏向行首，剩下的区别是输出顺序：
  aiter / vLLM 自带按扫描顺序（位置大致升序），Triton 按候选到达顺序。加了实验开关 `VPD_TOPK_SORT=1`（select 后按位置升序排序），
  待名额空出：先看各实现的输出顺序（309093，`scripts/topk_output_order.py`），再用 a17（只门禁，Triton + 排序 vs vLLM 自带，交替各两次）验证。
  计分用模拟接受，不受影响；真实部署里 −1.2% 的接受长度约等于 −1% 吞吐，比 top-k 的收益小，但要弄清楚。
- **同分值时各实现取哪些位置**（309005，`results/topk-ties/`）：aiter、vLLM 自带、修复版取的 0 都偏向行首（700K 行开头 10% 的占比 1.00 / 0.58 / 1.00），
  没有谁偏向行尾，接受长度的约 1% 差别不是这个原因，暂按噪声处理。
- **a12 被取消（09:52，状态 CANCELLED，不是我、也不是另一个会话发的；v3-120 随即被别的作业占用）**。取消前调优已写出 32 行
  （`results/a12/partial-hipb-tuned-32rows.csv`）：加了 hipBLASLt 后 o_proj 在 M = 56 / 64 / 72 是 39.6 / 40.5 / 44.1 µs，a4 的 FlyDSL 是 40.9 / 41.2 / 45.7，
  每次只快 1–1.6 µs（每步约 0.1 ms）；q_b 新结果（M = 64 时 16.0）比镜像 llama405B 表里的 13.0 还慢。**dense GEMM 已接近现有 kernel 的上限，a12 不再重跑**。
- **a13（308658，不钉节点，`once/p4d4-pair2b.sh`，只做门禁）**：同一作业里 a8 配置与 + Triton top-k 的门禁交替各两次（A1 B1 A2 B2），量真实接受长度的波动，
  判断 a11 的 3.41 是不是 Triton top-k 造成的。`p4d4-pair2b.sh` = `p4d4-pair2.sh` + 重复编号（`PAIR_SIDES` 里的 A1 / B2 等）+ `S_SCORE=0`（只门禁）。
- **a12（308490，钉 v3-120，排在 a11 后，`once/p4d4-a12-gemmhipb.sh`）**：dense GEMM 带 hipBLASLt 解重调（a4 为省时去掉了），60 个形状 = o_proj、q_b、fused_qkv_a、
  indexer wq_b × 目标模型 FULL graph 的 M（32–128 步长 8、136、144）。a8 trace 里 o_proj 约 3.5 ms / 步（41–47 µs，带宽下限 33.5 µs），fused_qkv_a
  约 0.9 ms（11.5–12.7 µs，镜像 glm5 表里的 FlyDSL），q_b 约 0.8 ms。aiter 合并多份表时同一形状出现两次会改写源文件并报错（只读挂载会失败），
  所以 `scripts/merge_tuned_gemm2.py` 按形状取快者：镜像表里没有的（o_proj）进我们的表，已有的写成那份镜像表的修改副本、原路径挂载（同键替换）。
  调优用一次性 JIT 卷，调完删掉。之后用 `p4d4-pair2.sh` 只跑 B 侧（门禁 + 计分，`S_GEMM_DIR`），是否带 Triton top-k 按 a11 结果定。
- **v3-186 被 drain**（05:44，`launch failed: prepare stepd directory: No space left on device`，a9 刚结束）：节点本地盘满。可能的原因之一是 a4 的
  GEMM 调优把数十万个 FlyDSL 候选编译进了节点本地的 docker 卷 `vpd-aiter-jit-*`（serving 也挂这个卷）；节点 drain 后我无法登录清理，需要管理员处理。
  之后的轮次换节点，换节点后先用 a8 配置重新打一次基线再做 A/B。以后调优改用一次性的 JIT 卷，调完删掉。

## 0.0001 a8 成绩：53,702 tok/s/GPU，超过 sglang P4D4 的 53,351（100.7%）（a7 + a6 的改动，11-24 04:10，单次）

307733，v3-186，`once/p4d4-score2.sh` + `once/score-a8.env`，结果 `results/a8/`。配置 = sync skip + decode 前缀缓存 + DP 引擎均衡（KV 权重 14，容量 3,141,120）
+ GEMM 表 + FP4 dispatch + decode `--max-num-batched-tokens 2048`。没有单独做门禁：FP4 dispatch 的覆盖层与 a7 过门禁时相同，mnbt 不改变数值
（汇总 token > 2048 的块仍走 BF16 分发）。

| | a6（KV） | a7（FP4） | **a8（两者）** | sglang P4D4 C104 |
|---|---|---|---|---|
| tok/s/GPU | 53,312 | 53,092 | **53,702** | 53,351 |
| 稳态 300 s 起 / 300–1800 s | 55,962 / 57,073 | 55,720 / 57,521 | **56,394 / 57,992** | — |
| req/s / 完成 / 丢弃 | 3.520 / 12,755 / 7 | 3.519 / 12,766 / 7 | 3.543 / 12,849 / 6 | 3.469 / 12,582 / 1 |
| 交互 p50 / p90 | 63.1 / 58.5 | 64.5 / 60.2 | 64.7 / 60.4 | 69.9 / 60.8 |
| ITL p50 / p90（ms） | 15.8 / 17.1 | 15.5 / 16.6 | **15.5 / 16.6** | 14.3 / 16.4 |
| TTFT p50 / p90 / 均值（s） | 2.52 / 6.76 / 3.85 | 2.69 / 7.76 / 4.20 | **2.53 / 6.63 / 3.83** | 4.20 / 9.23 / 5.65 |
| 某 rank KV > 95%；decode 等待 p50 / p90 | 35%；2 / 7 | 49%；3 / 9 | 35%；2 / 7 | — |
| DP 不均代价（ms / 步） | — | 1.66 | 1.51 | — |
| hidden all-gather p10 / p25 / p50（µs） | — | 4.8 / 5.0 / 8.4 | 5.0 / 5.1 / 9.2 | — |

- 两项改动作用在不同环节、效果叠加：KV 侧与 a6 相同（KV 满的时间 35%、TTFT p90 6.63 s），步长侧与 a7 相同（ITL 15.5 ms、hidden all-gather 约 5 µs）。
  测量期第 5–25 min 的 decode 生成速度（`scripts/decode_gen_rate.py`，4 个引擎合计）：a5 4,096 / a6 4,118 / a7 4,167 / **a8 4,189** tok/s。
- 单次；a9（307883）同配置同节点重复，04:13 起，用来给出单次波动。
- 与 sglang 比：吞吐、TTFT（−40%）、TTFT p90 更好；ITL 仍慢约 8%（15.5 对 14.3 ms），交互 p50 低（64.7 对 69.9）。

## 0.0003 a6 成绩：53,312 tok/s/GPU（a5 + decode `--max-num-batched-tokens 2048`，同节点比 a5 +1.0%，11-24 02:38，单次）

307324，v3-186，`once/p4d4-score.sh` + `once/score-a6.env`，结果 `results/a6/`。不含 FP4 dispatch，与 a5 单因素比较。

| | a5 | **a6（mnbt 2048）** | a7（a5 + FP4） |
|---|---|---|---|
| tok/s/GPU | 52,784 | **53,312** | 53,092 |
| 稳态 300 s 起 / 300–1800 s | 55,398 / 56,852 | 55,962 / 57,073 | 55,720 / 57,521 |
| decode KV 容量 / rank（token） | 2,955,712 | **3,141,120（+6.3%）** | 2,955,712 |
| ITL p50 / p90（ms） | 15.8 / 16.9 | 15.8 / 17.1 | 15.5 / 16.6 |
| TTFT p50 / p90 / 均值（s） | 2.61 / 7.92 / 4.17 | **2.52 / 6.76 / 3.85** | 2.69 / 7.76 / 4.20 |
| 某 rank KV > 95% 的采样；decode 等待 p50 / p90 | 47%；3 / 10 | **35%；2 / 7** | 49%；3 / 9 |
| decode 排队时间（engine 指标） | 1.16 s | 0.85 s | — |

- **sglang 53,351 的 99.9%**。激活预留从 8192 token 降到 2048，KV 多 6.3%（估计是 +4%），步长不变（FULL 中位按最忙 rank 请求数与 a5 相同），
  收益全在 KV：KV 满的时间 47% → 35%，decode 排队 −0.3 s，TTFT p90 −1.2 s。一小时内 decode 上 > 16 token 的块共约 2.7 万 token（4 个 rank），
  2048 的块上限没有造成问题；没有崩溃（m9b 那次 1024 的崩溃已在主线归因为显存余量，与此无关）。
- vLLM 对重复的 `--max-num-batched-tokens` 取最后一个（日志 "Found duplicate keys"，引擎配置 `max_num_batched_tokens: 2048`）。
- a7（FP4，步长）与 a6（KV）作用在不同环节，**a8 = a7 + a6 的改动**（`once/score-a8.env`，KV 均衡的容量参数同步改成 3,141,120），02:40 起。

## 0.0005 a7 成绩：53,092 tok/s/GPU（a5 + FP4 dispatch，同节点比 a5 +0.6%，11-24 01:07，单次）

307524，v3-186，`once/p4d4-a7-fp4dispatch.sh`，结果 `results/a7/`。门禁见 0.003 节（干净，接受 3.477）。

| | a5 | **a7（+ FP4 dispatch）** | sglang P4D4 C104 |
|---|---|---|---|
| tok/s/GPU | 52,784 | **53,092** | 53,351 |
| 稳态 300 s 起 / 300–1800 s | 55,398 / 56,852 | 55,720 / 57,521 | — |
| req/s / 完成 / 丢弃 | 3.502 / 12,701 / 5 | 3.519 / 12,766 / 7 | 3.469 / 12,582 / 1 |
| ITL p50 / p90（ms） | 15.8 / 16.9 | **15.5 / 16.6** | 14.3 / 16.4 |
| TTFT p50 / p90 / 均值（s） | 2.61 / 7.92 / 4.17 | 2.69 / 7.76 / 4.20 | 4.20 / 9.23 / 5.65 |
| FULL 步中位，按最忙 rank 请求数 @16 / @18 / @22 / @26（ms） | 45 / 48 / 50 / 52 | 45 / 47 / 49 / 51 | — |
| DP 不均代价（ms / 步） | 1.87 | 1.66 | — |
| hidden all-gather（dp1 trace，µs）p10 / p25 / p50 | 19.6 / 23.6 / 28.7（a3） | 4.8 / 5.0 / 8.4（FP4 + scale 两次） | — |

- **sglang 53,351 的 99.5%**。+0.6% 在单次波动内，但机制上的证据一致：trace 里 hidden 的 all-gather 从约 20–29 µs 降到约 5–8 µs
  （现在每层两次：FP4 和 scale，`results/a7/kernel-dist-dp1.txt`），同负载 FULL 步短 1–2 ms（负载越大省得越多，符合"数据量越大越省"），ITL −0.3 ms。
  MoE 量化 kernel 仍是每层一次（现在只量化本 rank 的行，4.6 µs），`mxfp4_moe_sort` 照旧。
- 数值：GSM8K 197/200、真实接受 3.477，与不开时相同；aiter 的 per-1×32 量化在分发前后是同一个函数，按行计算。

## 0.001 a5 成绩：52,784 tok/s/GPU（a3 + a4 的 GEMM 表 + KV 感知 DP 均衡 W = 14，同节点比 a3 +1.2%，11-23 23:21，单次）

307068，v3-186，`once/p4d4-a5-dplbkv.sh`，结果 `results/a5/`（`lb-kv-active.txt` 确认 decode API server 加载了 `[vpd] DP LB KV weight 14.0`）。

| | a3 | **a5** | sglang P4D4 C104 |
|---|---|---|---|
| tok/s/GPU | 52,144 | **52,784** | 53,351（不含 PTPC 约 52.5k） |
| 稳态 300 s 起 / 300–1800 s | 54,699 / 56,107 | 55,398 / 56,852 | — |
| req/s / 完成 / 丢弃 | 3.471 / 12,590 / 11 | 3.502 / 12,701 / 5 | 3.469 / 12,582 / 1 |
| ITL p50 / p90（ms） | 15.6 / 17.0 | 15.8 / 16.9 | 14.3 / 16.4 |
| TTFT p50 / p90 / 均值（s） | 2.76 / 9.27 / 4.68 | **2.61 / 7.92 / 4.17** | 4.20 / 9.23 / 5.65 |
| 最忙 rank running p50 / p90；最忙 / 平均 | 16 / 22；1.13 | 17 / 22；1.18 | — |
| 某 rank KV > 95% 的采样；decode 等待 p50 / p90 | 53%；4 / 11 | 47%；3 / 10 | — |
| 抢占 | 20 | 10 | — |
| DP 不均代价（ms / 步；请求数 + 上下文，测量期第 1–21 min） | 1.86（1.02 + 0.83） | 1.87（1.45 + 0.43） | — |

- **sglang 53,351 的 98.9%，超过不含 PTPC 的同口径目标 52.5k（100.5%）**。300 s 起的稳态和 300–1800 s 两个口径也都是 +1.2–1.3%。
- 收益来自 KV 侧，不是步长：KV 权重把上下文的不均从 0.83 压到 0.43 ms，但请求数的不均从 1.02 涨到 1.45 ms，合计不变（FULL 步中位都是 47 ms、
  ITL 持平）；KV 满的时间少了（53% → 47%），decode 排队少了，抢占减半，TTFT p90 −1.35 s，所以同样的步长下完成的请求更多。
  只按到达选 rank 时，不均的下限约 1.8 ms / 步（请求的结束是随机的，路由管不到）。
- 归因：a5 相对 a3 同时加了 GEMM 表（a4 卡死前同窗口 +1.4%，在噪声内）和 KV 权重，两者没有分开；单次。之后的 a7、a6 都在 a5 的配置上叠加。

## 0.002 a4 结果：计分作废（prefill 在测量第 46 min 卡死），卡死前的窗口比 a3 +1.4%（11-23 22:00，单次、部分）

- 门禁干净（`results/a4/gate.txt`）：smoke 6/6，GSM8K 195/200，乱码 0，16k 32/32、长前缀 31/32，真实接受 3.461；decode 日志里新表在 aiter 的合并列表中，
  M ≤ 256 除 lm_head 外不再有 "not found tuned config"。
- **计分 46,023，rc = 1（1,126 个错误），不算成绩**：21:41 起 **prefill**（不是 decode）停住，看门狗 21:43:59 报"progress 180 s 不变"并抓了 py-spy
  （`runs/p4d4-a4-gemmtune-dprefix-syncskip-3600/hang/`）：PP2、PP3 两个 worker 的主线程都停在 CPU 层回读 KV 的 `swap_blocks_batch`
  → `hipMemcpyBatchAsync`（HSA 运行时内部），21:45:49 EngineCore 报 `RPC call to sample_tokens timed out` 退出。与 a4 的改动无关（GEMM 表只挂在 decode 上）；
  与主线 s2c 那次 prefill 卡住（同样怀疑 offload 的批量拷贝，之后加了 `VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS=512`，本轮也带着）是同一类，偶发
  （a0–a3 四轮、主线 m10 等都没有）。dmesg 里 21:32 起的 ionic `IONIC_CMD_LIF_SETPHC ... TIMEOUT` 在 a3 开跑前就有 47 条，a3 没出事，不当原因。
- **卡死前的同窗口比较**（两轮的数据集顺序相同，每 300 s 窗口的请求数和 ISL 几乎一致，`results/a4/window_tput.txt` 对 `results/a3/`）：
  300–1800 s 56,728 对 56,107（+1.1%），300–2700 s 八个窗口平均 54,689 对 53,952（+1.4%）；单个窗口两轮之间差 ±4%，所以这 +1.4% 在噪声内。
  ITL p50 / p90 15.8 / 16.9 对 15.6 / 17.0 ms；测量期第 25–43 min 的 FULL 步长按最忙 rank 请求数与 a3 相同（@16 45 对 46 ms，@17 48 对 47 ms）。
- 结论：表是对的（门禁干净、隔离测时在常用 M 上快 4–5 µs），端到端收益 ≤ 1%、单次测不出来；**保留**（a5、a6 都挂着它）。

### a7 设计：MoE 分发前先量化成 MXFP4（FP4 dispatch，11-23 22:10）

- **依据**：decode 的 MoE 走 naive DP 分发（all-gather BF16 hidden，12,288 B / token），aiter 的 fused_moe 再在汇总后的行上量化成 MXFP4（per 1×32）。
  量化按行做，所以各 rank 先量化自己的行再 all-gather，字节相同、数据量约 1/4（3,072 B 打包 FP4 + 192 B e8m0 scale）。a3 trace（dp1）里
  BF16 hidden 的 all-gather p10 / p25 / p50 = 19.6 / 23.6 / 28.7 µs（低分位≈纯传输，最忙的 rank 最后到、只付这部分），4 字节小 all-gather 稳定约 5 µs
  （`scripts/kernel_dist.py`）。预计每层省约 8–11 µs、每步约 0.6–0.9 ms（约 1.5%）。
- **实现**（`scripts/make_fp4dispatch_overlay.py`，两份覆盖层，`VPD_FP4_DISPATCH=1` 才生效）：照搬 MoRI 的 FP4 分发
  （`prepare_finalize/mori.py:98-105`，`get_hip_quant(QuantType.per_1x32)(a1, shuffle=False)`，scale 不重排），aiter `fused_moe_2stages` 有对应分支
  （"Input is already quantized to fp4x2 (e.g., from FP4 dispatch), skip re-quantization, only sort the scale"）。FP4 和 scale 以 BF16 视图 all-gather，
  仍走 aiter 自定义 all-gather。`AiterExperts.workspace_shapes` 按分发宽度定输出宽度，打包时是 hidden 的一半，同 MoRI 一样乘 2。
- **限制**：只在汇总后 128 < token ≤ 2048 时启用。aiter 按下一个 2 的幂选 MoE kernel，≤ 128 用的 block_m 16 kernel（`_f16in`）直接读 BF16、
  给它预量化输入会非法访存（aiter 源码注释）；256–2048 是调过的 block_m ≥ 32 FP4 激活 kernel。只对 MXFP4 W4A4 专家（MTP 的 BF16 MoE 不受影响）、无 LoRA。
- **验证**：作业开头 `scripts/test_fp4dispatch.py`（1 GPU：量化输出 dtype / 形状、BF16 视图往返逐位一致、`workspace_shapes`）；门禁（真实接受）干净才计分。
- **排队**：a7（307524）排在 a6 前面（a6 先 hold、a7 开始后自动 release）：a7 收益更大、风险也更大，门禁不过的话 30 min 内就退出，还有时间修。
- **a7 进展（11-23 23:40）**：基础配置 = a5（KV 权重 14 + GEMM 表，`once/a7-extra.env`）。作业开头的单元检查通过（`results/a7/overlay-build-test.txt`：
  aiter per-1×32 量化输出 `float4_e2m1fn_x2 (M, 3072)` + `float8_e8m0fnu (M, 192)`，BF16 视图往返逐位一致；`workspace_shapes` 打包时输出宽度 6144）。
  **门禁干净**：smoke 6/6，GSM8K 197/200（a2 197、a4 195），乱码 0，16k 32/32、长前缀 31/32，真实接受 3.477（a4 3.461）；decode 日志无报错。
  23:38 起 3600 s 计分（第 1500 s 抓 trace，用来确认 hidden all-gather 变短）。

### DP 不均的代价（`scripts/dp_imbalance.py`，11-23 22:05）

按 a3 步长拟合的系数，每步每个 rank 的代价 = 0.74 × 请求数 + 3.9 × 上下文（M），把 4 个 rank 的 FULL 记录按时间配成全局步，算"代价最大的 rank − 平均"：

| | 每步多出（ms） | 其中请求数部分 | 其中上下文部分 | 请求数 最大 − 平均 | 上下文 最大 − 平均 |
|---|---|---|---|---|---|
| a2（路由选 rank） | 2.95 | 2.46 | 0.49 | 3.49 | 0.24 M |
| a3（vLLM 按 waiting + running 均衡） | **1.86** | 1.02 | 0.83 | 1.73 | 0.30 M |

- a2 → a3 少了约 1.1 ms / 步（约 2.3%），与 a3 的 ITL −0.6 ms 同向。a3 的均衡把请求数摊平了，但上下文的不均反而变大（0.49 → 0.83 ms）。
- 完美均衡最多还能省约 1.9 ms / 步（约 4%）；KV 感知均衡（a5）针对的是上下文那部分（约 0.8 ms，约 1.8%），现实能拿到的估计在 1% 左右。
  a5 的权重 14：a3 里上下文的 最大 − 平均 0.30 M ≈ 0.10 的 KV 用量 × 14 ≈ 1.4 个请求，和请求数的 1.7 同量级，两项大致等权。

## 0.003 a4 / a5 进行中（11-23 19:10）

- **a4（306878，v3-186，18:38 起）= a3 + decode BF16 GEMM 调优表**，计分时 `INFERA_VLLM_DP_ENGINE_LB=1`（`$R/once/a4-extra.env`）。
- **调优中途干预（18:48）**：按 a2 decode 日志列出的 256 个形状、`--libtype all --with-hipblaslt` 跑，每个形状约 43 s（hipBLASLt 5,943 个解 +
  FlyDSL 约 1,000–3,800 个候选），256 个要约 3 h；而调优器每 100 个形状才写一次表（`--batch` 默认 100），75 min 超时前最多写出第一批，
  第一批里大半是用不到的大 M 形状。于是把 `runs/a4-gemmtune/untuned.csv` 换成 141 个 decode 计时相关的形状
  （`gemm-tune/glm52_decode_shapes_curated141.txt`：o_proj 6144 × 16384 的 47 个 M ≤ 256 排最前，然后 dense MLP gate_up 24576 × 6144 的 45 个、
  q_b 16384 × 2048 的 2 个、MTP 共享专家 gate_up 4096 × 6144 的 47 个；去掉 lm_head 的 89 个——带宽受限、约 346 µs / 次，已到 5.5 TB/s——和 M > 256
  的重算形状），再停掉调优容器，让 `tune_decode_gemm.sh` 走重试分支（`--libtype all`，不含 hipBLASLt；候选里有 `torch` 即当前默认路径，所以不会更慢）。
  第一批 100 个 19:37 写出；第二批也在 20:13 写完，共 141 行（`gemm-tune/glm52_decode_bf16_tuned_gemm.csv`，a4 挂的就是这份）。
  **超时没生效**：`timeout` 只给 docker 客户端发 SIGTERM，容器里 PID 1 是 bash、不处理信号，调优多跑了约 10 min（作业时长仍够）。
  `tune_decode_gemm.sh` 已改成 `docker run --init` + `timeout -k 60`、默认 `--batch 16`、去掉重试分支（之后的调优用新版）。
  o_proj 是重点：a1 trace 里每步约 81 次（目标 78 层 + 草稿 3 次）、每次约 46.5 µs（hipBLASLt 默认 `MT128x80x128` / `MT192x96x128`），
  按 201 MB 权重的带宽下限约 33.5 µs，最多省约 1 ms / 步。
- **调优结果（隔离测时，单位 µs）**：o_proj 在常用的 M = 56–64（每 rank 14–16 个请求 × 4）是 FlyDSL 40.9–41.2，torch 默认在相邻的 M = 68 / 76
  是 45.7 / 46.1（trace 里默认约 45），**每次约省 4–5 µs、每步约 0.3–0.4 ms（< 1%）**；M ≤ 64 都是 FlyDSL（35.7–41.2，约 4.9–5.6 TB/s），
  M = 136–208 多数仍是 torch。dense MLP gate_up 在 M ≤ 64 全是 torch（默认就是最优），q_b 两个 14.0，MTP 共享专家 gate_up 11.1–24.4（FlyDSL）。
  所以 a4 的预期收益在单次波动内；没调 hipBLASLt 解（第一次全量调优停掉了），o_proj 离带宽下限还有约 7 µs / 次。
- **a5（307068，排在 a4 后）= 最优配置 + KV 感知的 DP 均衡**（`scripts/make_dplb_kv_overlay.py`，测试 `scripts/test_dplb_kv.py` 在镜像里通过）。
  v0.34.0 的均衡打分 `max(inflight, waiting + running)`，只在有 waiting 时按 KV 加罚；a3 的步长拟合 `26.7 + 0.74 max_req + 3.9 max_ctx(M)`
  说明慢 rank 由请求数和上下文共同决定，一个 rank 的 KV（2,955,712 token）满了约等于 15.6 个请求。覆盖层加 `VPD_DP_LB_KV_WEIGHT × kv_cache_usage`
  （a5 用 14），并在协调器 100 ms 快照之间把所选 rank 的用量加上该请求的 prompt / 容量。只改路由，不做门禁。a3 里最忙 / 最空 rank 的 KV 中位
  0.96 / 0.70，相当于约 3.6 个请求的差。是否带 a4 的 GEMM 表由 `$R/once/a5-extra.env` 在计分前决定。

## 0.004 a3 成绩：52,144 tok/s/GPU（a2 + decode DP 引擎均衡，同节点比 a2 +1.1%，11-23 18:36，单次）

306707，v3-186，`once/p4d4-a3-dplb.sh`（a2 配置 + `INFERA_VLLM_DP_ENGINE_LB=1`，只计分不门禁：只改路由，不碰计算）。结果 `results/a3/`。

| | a2 | **a3（+ DP 均衡）** | sglang P4D4 C104 |
|---|---|---|---|
| tok/s/GPU | 51,572 | **52,144** | 53,351（不含 PTPC 约 52.5k） |
| 稳态 300 s 起 / 300–1800 s | 54,120 / 56,250 | 54,699 / 56,107 | — |
| req/s / 完成 / 丢弃 | 3.411 / 12,377 / 8 | 3.471 / 12,590 / 11 | 3.469 / 12,582 / 1 |
| ITL p50 / p90（ms） | 16.2 / 17.9 | **15.6 / 17.0** | 14.3 / 16.4 |
| TTFT p50 / p90 / 均值（s） | 2.64 / 7.59 / 4.10 | 2.76 / 9.27 / 4.68 | 4.20 / 9.23 / 5.65 |
| 最忙 rank running p50 / p90；最忙 / 平均 p50 | 18 / 26；1.28 | **16 / 22；1.13** | — |
| decode KV 最大 p50；某 rank > 95% 的采样 | 0.94；42% | 0.96；53% | — |

- 均衡生效：最忙 rank 的请求数中位从 18 降到 16（平均每 rank 14–14.5 不变），FULL 步中位 48 → 47 ms，ITL p50 −0.6 ms（−3.7%）。
  主线 m14（同开关，1800 s）也是 +1.3%，两次同向；但 +1.1% 仍在单次波动内，**待重复**。
- 代价：KV 更满（> 95% 的采样 42% → 53%），decode 等待略多，TTFT p90 7.59 → 9.27 s；吞吐仍净增。
- trace（1500 s，dp0）：GPU 忙碌 98.3%，每步规律空闲 0.84 ms（含一次 34 ms 的偶发空隙）。
- **sglang 53,351 的 97.7%**（不含 PTPC 的 52.5k 的 99.3%）。a4（306878，18:38 起）= a3 + decode BF16 GEMM 调优，计分时 DP 均衡保持打开。
- MoE 调优表核对（18:40）：主模型的 MXFP4 MoE（257 专家 top-9）全部命中 aiter 调优表，两段 kernel 已到 5.4–6.0 TB/s。
  MTP 层是未量化的 BF16（256 专家 top-8），decode 日志里是 `[fused_moe] using 2stage default`；a1 trace（dp1，62 步）里每步 3 次、
  合计约 2.1 ms。但草稿第 2、3 步每次约 64 个 token × top-8，256 个专家里约 220 个被激活，读约 4.2 GB，用时 0.62 ms（约 6.8 TB/s），
  第 1 步约 5.8 TB/s，已接近带宽上限（计划 2.3 节的判断），调优空间约 0.1–0.2 ms / 步，不单独排作业；量化草稿 MoE 被 InferenceX Check 13 禁止。

## 0.005 a2 成绩：51,572 tok/s/GPU（sync skip + decode 前缀缓存，同节点比 a1 +4.2%、比 a0 +13.8%，11-23 17:04，单次）

| | a0 | a1（+ sync skip） | **a2（+ decode 前缀缓存）** | sglang P4D4 C104 |
|---|---|---|---|---|
| tok/s/GPU | 45,337 | 49,476 | **51,572** | 53,351 |
| req/s / 完成 / 丢弃 | 3.073 / 11,152 / 8 | 3.357 / 12,181 / 7 | 3.411 / 12,377 / 8 | 3.469 / 12,582 / 1 |
| 交互 p50 / p90 | 59.2 / 49.9 | 61.2 / 50.3 | 61.8 / 56.0 | 69.9 / 60.8 |
| ITL p50 / p90（ms） | 16.9 / 20.1 | 16.3 / 19.9 | 16.2 / 17.9 | 14.3 / 16.4 |
| TTFT p50 / p90 / 均值（s） | 2.99 / 13.14 / 6.56 | 2.72 / 9.57 / 4.60 | **2.64 / 7.59 / 4.10** | 4.20 / 9.23 / 5.65 |
| 重算块超出时间（steplog） | 10.8% | 6.6% | **0.2%** | — |
| decode KV p50 / p90；某 rank > 95% 的采样 | — | 0.86 / 0.99；57% | 0.83 / 0.97；42% | — |

- sglang 53,351 的 96.7%（不含 PTPC 的同口径约 52.5k 的 98%）。门禁干净（197/200、32/32 + 31/32、接受 3.455）。
- 17 次抢占只重算约 1.1 万 token（被抢占的请求命中自己尚未复用的块）；FULL 步中位仍 48 ms。trace 里规律性的每步空闲约 0.05 ms，
  4 份 trace 里各有一次 80–345 ms 的单次空隙（偶发）。
- **剩下的差距在 decode 每个 token 的耗时**（TTFT 已好于 sglang）：同负载 FULL 步 vLLM @16 45 ms 对 sglang 41 ms；a1 trace 里多出的是
  dense GEMM（+2–4 ms）和 MoE all-gather 等最慢 rank（约 +3 ms）。下一步：a3 = a2 + decode DP 均衡（等待部分）；之后 dense GEMM 调优。

## 0.01 a1 成绩：49,476 tok/s/GPU，同节点比 a0 +9.1%（11-23 15:19，单次）

| | a0（v3-186，m10 配置） | **a1（v3-186，m10 配置 + sync skip）** | m10（v3-006） | sglang P4D4 C104 |
|---|---|---|---|---|
| tok/s/GPU | 45,337 | **49,476（+9.1%）** | 48,572 | 53,351 |
| req/s / 完成 / 丢弃 | 3.073 / 11,152 / 8 | 3.357 / 12,181 / 7 | 3.297 / 11,951 / 8 | 3.469 / 12,582 / 1 |
| 交互 p50 / p90 | 59.2 / 49.9 | 61.2 / 50.3 | 59.4 / 48.5 | 69.9 / 60.8 |
| ITL p50 / p90（ms） | 16.9 / 20.1 | 16.3 / 19.9 | 16.9 / 20.6 | 14.3 / 16.4 |
| TTFT p50 / p90 / 均值（s） | 2.99 / 13.14 / 6.56 | 2.72 / 9.57 / 4.60 | 2.88 / 10.97 / 5.20 | 4.20 / 9.23 / 5.65 |
| decode 抢占（整轮） | 20 | 26 | 22 | — |

- 3600 s 口径的新最好成绩，sglang 53,351 的 92.7%（不含 PTPC 的同口径约 52.5k 的 94%）。同节点、同配置只差补丁；a0 多一次 trace（各有一次 py-spy），
  扰动略大于 a1。以前同节点同配置两轮差 0.8–2%（m10 / f1、m10 / m12 前段），9% 应是真实收益，但两边都是单次。
- 抢占重算（H2）仍在：a1 测量期前 30 min 重算块的超出时间 10.8%。a2 = a1 + decode 前缀缓存，同节点接着跑。

## 0.015 sync skip 补丁与上游的关系（11-23 14:55 查重）

- **#58710（open，外部贡献者）"[ROCm][Perf] Skip the seq_lens D2H sync in the AITER sparse MLA metadata key"** 做的就是我们补丁的第一部分：
  用 `seq_lens_cpu_upper_bound` 判断截断到 `index_topk` 是否已饱和，饱和时跳过 `.cpu()`（他们的余量是 `num_speculative_tokens`，我们取 32）。
  作者没有 ROCm 硬件，PR 写明"Needs validation on ROCm hardware"。按 vLLM 查重规则不另开 PR，**留言稿给 #58710 附 MI355X 实测**（a1 成绩出来后写，不发）。
- 补丁第二部分（关掉指纹变化后的 `synchronize()`）：这次同步是 **#47404**（07-09 合并，修 graph 回放读到未写完的持久化元数据）加的，
  **#66400（11-09 合并）已在 main 上删掉**（它违反 `VLLM_GPU_SYNC_CHECK=error`，并附了测试）。v0.34.0 还带着它；我们在 v0.34.0 上去掉，与 main 现状一致，门禁干净。
- 相关的还有 #67232（修 decode 路径上的一处同步等）、#61504（纯 decode 时在主机上建 indptr），都是 open。
- 所以 P4D4 迁移到 main 时，需要的是 #58710 合入（#66400 已在 main）；在那之前用本目录的覆盖层。

## 0.02 a1（306294，v3-186，sync skip 补丁）门禁与中途结果（11-23 14:25）

- **门禁干净**（真实接受，CPU 层 200 GB）：smoke 6/6，GSM8K 198/200（乱码 0、开头坏 0），16k 捞针 32/32、长前缀 32/32，真实接受长度 3.458。
  补丁不影响正确性。14:15 开始 3600 s 计分。
- **py-spy（14:21，用 `srun --jobid=306294 --overlap` 跑 `scripts/pyspy_decode.sh`）**：decode worker 主线程只有约 4% 在执行 RPC，36% 在
  `MessageQueue.dequeue` 等下一步的调度；输出线程 50% 在 `Event.synchronize` 等 GPU；EngineCore 仍 97% 在等 worker 的输出。
  打补丁前 worker 93% 卡在 `seq_lens.cpu()`。等待点从"主机等同步"变成了"等 GPU"。
- **trace（测量期 1500 s，4 个 rank）：H3 解决了**。GPU 忙碌 99.7%（a0 是 83–92%），3.1 s 里只空闲 9–10 ms，每步约 0.15 ms（a0 每步 4–8 ms，
  sglang 0.2–0.8 ms）；每次 graph 提交时主机领先 GPU 49–82 ms（中位 76 ms，a0 是 0.06 ms），阻塞的 `hipMemcpyWithStream` 没有了。
  现在每步时间 ≈ GPU 计算时间（这个窗口约 49.9 ms / 步，62 步）。按类（ms / 步）：MoE GEMM 17.8、dense GEMM 11.8、通信 8.5、
  moe_misc 3.6（含 indexer top-k 1.5）、sparse MLA 3.2、indexer 打分 1.5、norm 1.4、其他 2.2。与 sglang TP4 C104 verify 比：MoE 持平，
  dense 多 2–4（vLLM 含 4 次 lm_head 与 router gate），**通信多约 3.1**（bf16 all-gather 每次 69 µs，聚合模式下 34.5 µs，多出的是等最慢的 rank），
  注意力少约 4。`results/a1/` 里的 trace-gaps 由作业结束时生成。
- **FULL 步长（同节点，按最忙 rank 请求数，补丁 a1 对 a0；a1 只有测量期第 5–7.5 min 的 2,517 步）**：
  10 个 38 对 42 ms，13 个 42 对 47，16 个 45 对 47，18 个 47 对 49，20 个 48 对 49，25 个 52 对 54。小 batch 快 4–5 ms（约 10%），
  典型负载（16–18）快约 2 ms。测具里看不出来，是因为测具的 GPU 步长被常数假 KV 拉长到 74 ms，主机开销被掩盖了。

## 0.03 b0 第 3 轮（12:56–13:34）：去掉同步的补丁在测具里几乎没有效果

`results/b0/r3-syncskip-wm003.txt`。补丁生效（4 个 worker 全部构建都跳过了 seq_lens 同步，指纹重算后的 `synchronize()` 也关了），但 FULL 步
长按最忙 rank 请求数比：

| 最忙 rank 请求数 | 20 | 25 | 30 |
|---|---|---|---|
| r1 A / B（基线） | 73 / 73 | 76 / 76 | 77 / 77 ms |
| r3 A（sync skip） | 72 | 74 | 76 ms |
| r2 A（`--watermark 0.03`） | 72 | 75 | 77 ms |
| r3 B（sync skip + `--watermark 0.03`） | 72 | 75 | 76 ms |

- 只快 0–2 ms，在噪声量级，远小于生产 trace 里每步 4–8 ms 的 GPU 空闲。说明去掉这两处同步后，CPU 转而在别处等待
  （候选：ROCm 上 `hipGraphLaunch` 对仍在执行的同一个 graph 再次提交时阻塞主机；草稿 decode 的两步回放的是同一个 graph），H3 的根因还要再找。
  a1（生产 P4D4 + 补丁，带 trace）会直接给出打补丁后 GPU 空闲的分布；测具这一轮没有 trace。
- r3 A 侧抢占重算很重（4.1% 的步、超出 24.3%），B 侧（水位线）为 0，再次说明水位线有效、抢占过程随机性大。

## 0.04 b0 第 2 轮（12:18–12:55）：`--watermark 0.03` 与 decode 前缀缓存都在测具里消掉了 H2（各一次）

`results/b0/r2-wm003-pc.txt`，与第 1 轮 A/A 比（同一测具、同一回放；测具过载、客户端有 fd 失败，见 0.05，数字只作方向）：

| | 抢占 | decode 本地重算 token | 重算块的超出时间 | 平均步长 | 生成 tok/s | 失败请求 |
|---|---|---|---|---|---|---|
| r1 A / B（基线） | 24 / 19 | 183 万 / 221 万 | 12.2% / 15.2% | 93.3 / 96.4 ms | 2,102 / 2,052 | 1,449 / 1,611 |
| r2 A：`--watermark 0.03` | **0** | **0** | **0** | 80.2 ms | **2,545（+21–24%）** | 915 |
| r2 B：decode 前缀缓存 | 27 | **2.7 万**（每次只重算末尾 ≤ 2.8K token） | 0.3% | 83.4 ms | **2,511（+20–22%）** | 970 |

- 水位线让抢占消失（准入时留 3% KV 余量）；前缀缓存不减少抢占，但被抢占的请求重新调度时命中自己还没被复用的块，重算量降到原来的约 1%。
  两者的 FULL 步中位不变（74–76 ms），收益全部来自去掉重算块。
- 生产上的取舍：水位线的代价是准入等待（TTFT），前缀缓存没有准入代价，还可能让同会话续轮在同一 rank 上少拉 KV；两者都只改部署参数。
  下一步在 P4D4 上测（a1 之后）。
- 第 3 轮（12:56 起）补丁生效：4 个 worker 的日志都是 "20000 of 20000 builds skipped the seq_lens sync"。

## 0.05 b0 第 1 轮（A/A，11-23 11:38–12:18，v3-186）与测具的问题

- `results/b0/r1-aa.txt`。两侧 FULL 步 p50 / p90：A 76 / 94 ms，B 76 / 92 ms；重算大块的步占 1.9% / 2.3%，相对同负载 FULL 步多出 **12.2% / 15.2%**
  的时间（24 / 19 次抢占）。测具复现了 H2，而且抢占过程本身随机性不小（同配置两侧差 3 个百分点）。
- **测具与生产不一致的三处**（第 2、3 轮的 bash 脚本已在执行，回放客户端保持与第 1 轮相同、三轮之间可比；b0 结束后再修）：
  1. 回放容器的 nofile 软上限是默认的 1024：约 30% 的请求 `Too many open files` 失败；
  2. 开环按生产的 3.3 req/s 发送，decode 处理不过来：请求排几分钟（延迟 p50 约 170 s），最忙 rank 约 25 个请求（生产约 17）；
  3. `DecodeBenchConnector` 以常数填 KV（`fill_std=0`），indexer 的 top-k 全是平局（A 线测过 aiter top-k 在大量平局时明显变慢），
     同样最忙 rank 25 个请求，FULL 步测具 76 ms、生产 54 ms。
  第 3 轮比的是同等 batch 下 FULL 步的步长（host 同步开销），受这三处影响小；端到端以 a1 的 P4D4 为准。下次测具：客户端提 nofile、
  `fill_std > 0`、闭环或按 decode 容量降低到达率。
- 显存：测具里 KV 不经 GEM，vLLM 的可扩展 KV 池每 rank 45,397 块（131 GiB），torch reserved 最高 139–143 GiB，空闲最低 6.3–11.5 GiB。

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
  （11:50 起，第 3 轮现场生成的覆盖文件同时去掉 `get_mla_metadata_v1` 重算后的 `synchronize()`：dense AITER MLA 后端每次构建都调它且不同步，
  它只写之后同一条流上的 kernel 才读的显存；`VLLM_ROCM_SPARSE_MLA_META_SYNC=1` 恢复。b0 的 bash 脚本 11:38 已开始执行、不能再改，只改了它在
  第 3 轮才读取的 Python 生成脚本。测具不检查正确性，上生产前要过 GSM8K 门禁。）结果在 `results/b0/`。
