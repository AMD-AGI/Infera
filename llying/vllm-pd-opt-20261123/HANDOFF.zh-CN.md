# 交接：GLM-5.2 P4D4（Infera + vLLM）继续优化（2026-11-24 16:55 写）

接手的 agent 请先读本文，再读同目录 `README.zh-CN.md` 最上面的"总览"、"复盘"、"TTFT 拆解"、a18 / a19 两节。计划在 `../vllm-pd-opt-plan-20261123/PLAN.zh-CN.md`。

## 1. 目标与用户的要求

- 目标：在 `dev/pd_opt/vllm_glm52_agentx` 分支上继续推高 P4D4（MI355X，8 卡）AgentX C104 3600 s 的 **tok/s/GPU**，同时降低 ITL / 提高 interactivity。
  对齐 sglang P4D4（53,351）只是初步目标（用户 11-24 05:50）。
- 用户 14:05 选定的主攻方向：**TTFT 与 decode 准入**（不改数值的调度类改动优先）；dense 层在线 FP8 会改数值、涉及 InferenceX 合规，未获批准，**不要做**。
- 工作方式（沿用）：自主决策；后台任务必须挂监控（Shell 的 `notify_on_output`），及时发现失败；大内存 / 分析任务不在登录节点做（用 `srun --jobid=<自己的作业> --overlap` 或申请作业）；
  及时把实验报告、数据、可复现脚本提交到仓库；文档用中文，结论要有数据，单次运行要标"单次"。
- 规则：只用跑完即退的作业（`run_once.sbatch`），不用 hold 作业；提交前看 `squeue -u liyingli`；每个用户最多 2 个运行中、4 个已提交的作业
  （另一个会话的 DSV4.1 作业 `lyl-dsv41-*` 也算在内，不要动它的作业 / 节点）；`git pull --rebase` 后再 push；`git commit -s` 并带
  `Co-authored-by: Cursor <cursoragent@cursor.com>`；不要对外发上游评论 / PR（只写草稿）；**不要修改正在执行的 bash 脚本**（作业执行的是 `$R/once/` 下的副本，
  改之前按作业核对；同内容覆盖也尽量避免）；不要注入宿主机的 libionic。

## 2. 当前最好的结果（都是单次）

| 配置 | tok/s/GPU | 交互 p50 / p90 | ITL p50 / p90 ms | TTFT p50 / p90 / 均值 s | 作业 / 节点 |
|---|---|---|---|---|---|
| a8 配置（基线，三次：a8 / a9 / a10A） | 53,702 / 53,427 / 53,717，平均 53,615 | 约 64.7 / 60.5 | 约 15.5 / 16.6 | 约 2.5 / 6.7 / 3.8 | v3-186、v3-120 |
| a15 B：a8 + vLLM 自带 decode top-k（`S_TOPK_IMPL=native`） | 53,897 | 65.6 / 60.9 | 15.2 / 16.4 | 2.42 / 6.70 / 3.79 | v3-239 |
| **a18 B：a8 + Triton top-k 第三版（排序）** | **54,484** | 69.5 / 63.6 | **14.4 / 15.7** | 2.53 / 7.23 / 4.13 | v3-231 |
| **a19 B：a8 + prefill `--long-prefill-token-threshold 4096`** | **54,449** | 65.2 / 60.5 | 15.4 / 16.5 | **2.44 / 6.14 / 3.33** | v3-037 |
| sglang P4D4 C104（对标） | 53,351 | 69.9 / 60.8 | 14.3 / 16.4 | 4.20 / 9.23 / 5.65 | — |

- a8 配置 = sync skip 覆盖层 + decode 前缀缓存 + DP 引擎均衡 + KV 感知均衡（W 14）+ decode BF16 GEMM 表 + FP4 MoE dispatch + decode `--max-num-batched-tokens 2048`。
  复现见 README 总览的"复现 a8"。
- **Triton top-k 有未解决问题**：门禁真实接受长度四次都低约 1.2%（3.41 / 3.427 / 3.416 / 3.407；不开时 3.455–3.477），GSM8K 正常。原因未明，
  a21 用来定位（目标层 vs 草稿层）。在弄清之前**不要把 Triton top-k 放进推荐配置**；计分用模拟接受，不受影响。
- a18 的 Triton 第三版：`scripts/vpd_triton_topk_decode.py`（2048 段 × 4 采样、OVER 4、CAND 16,384、同分值保护、基数直方图慢路径、每行升序排序 `_sort_rows`），
  覆盖层 `scripts/make_tritontopk_overlay.py`（`VPD_TOPK_IMPL=triton|native`、`VPD_TOPK_SCOPE=all|target|draft`、`VPD_TOPK_SORT=0` 关排序）。

## 3. 进行中 / 排队的作业（16:55）

| 作业 | 内容 | 状态 / 下一步 |
|---|---|---|
| 309187 a18 | `p4d4-pair2.sh` SCORE_ENV=a18：B = Triton 第三版（已完成 54,484），A = vLLM 自带 top-k，v3-231 | A 侧 15:59:53 开始测量，约 17:02 出分。结果在 `results/a18/A/`；记录"同节点 Triton vs native"的差值 |
| 309209 a19 | `p4d4-pair3.sh` SCORE_ENV=a19：B = long-prefill-threshold 4096（已完成 54,449），A = a8 配置，v3-037 | A 侧 16:16:52 开始测量，约 17:19 出分。结果在 `results/a19/A/`；记录成对差值 |
| 309212 a20 | `p4d4-pair3.sh` SCORE_ENV=a20 PAIR_SIDES=B：prefill `--max-num-batched-tokens 4096`，钉 v3-037 | a19 结束后开始；与 a19 A 侧比。结果 `results/a20/B/` |
| 309338 a21 | `p4d4-pair3.sh` SCORE_ENV=a21 PAIR_SIDES="A1 B1 A2 B2"，只门禁：A = Triton 只在目标层，B = 只在草稿层（MTP 层号 78） | 等名额。看 `results/a21/{A1,B1,A2,B2}/gate.txt` 的 `accept_len`：哪一侧降到约 3.41，问题就在那条路径 |

**本会话的监控进程会随会话结束，接手后请对每个作业重新挂监控**（见第 5 节）。

## 4. 接下来建议的顺序

1. 记录 a18 A、a19 A 的分数和成对差值（README 对应两节 + 总览表），提交。
2. a20 出分后判断 prefill 分块 4096 是否有效（看 TTFT 和 prefill 排队 / 运行时间：`results/a20/B/ttft_breakdown.txt`；分析脚本 `scripts/ttft_analyze.py RUN_DIR`）。
3. a21 出结果后：若只有草稿层降接受，可以只在目标层用 Triton（`VPD_TOPK_SCOPE=target`），先门禁确认接受长度恢复；否则继续查（同分值的取舍、输出与 aiter 逐行比对）。
4. **组合轮 a22**：a8 配置 + long-prefill-threshold 4096（+ prefill 分块 4096，若 a20 有效）+ top-k（Triton target-only 若接受正常，否则 vLLM 自带 native），
   `p4d4-pair3.sh` 成对（A = a8 配置）或在已有基线的节点上只跑 B 侧。之后对最好的配置重复一次给出波动。
5. TTFT 后续候选（README"TTFT 拆解"）：long-prefill 阈值扫 2048 / 8192、`--long-prefill-token-threshold-adaptive`；decode 侧排队 0.85 s
   （KV 拉取约 0.5 s：每次 Mooncake 约 1.33 GB / 3.3 万描述符 / 200 ms；等 KV 空间约 0.35 s）；prefill 的 CPU 层读回完成按步数发现（约 0.4 s / 请求）。
   每项先用"TTFT 每秒约 3% 吞吐"、"step 变化 × 0.3–0.6"估上限，低于 1% 的不单独占 P4D4。

## 5. 怎么做实验

- 工作目录 `R=/home/liyingli/bench_agentx/vllm-pd-20261117`（作业脚本、运行目录 `runs/`、日志 `logs/`、`logs/summary.out` 是所有轮次的分数汇总）；
  分支目录 `B=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52/llying/vllm-pd-opt-20261123`（脚本、结果、文档，提交这里）。
- 通用成对脚本 **`once/p4d4-pair3.sh`**（最新，支持 decode / prefill 参数、top-k、门禁、只门禁、重复编号），每侧配置 `once/score-<名>-{A,B}.env`，变量见脚本头注释。
  改好后拷到 `$R/once/`（确认没有作业正在执行同名文件）。提交示例：
  ```
  cd $R && X="$(paste -sd, /home/liyingli/bench_agentx/pd-repro-20261116/exclude_nodes.txt),crsuse2-m2m-v3-168,crsuse2-m2m-v3-186,crsuse2-m2m-v3-176,crsuse2-m2m-v3-113"
  sbatch -x "$X" -J lyl-vpd-aNN -t 04:50:00 --export=ALL,ONCE=p4d4-pair3,SCORE_ENV=aNN run_once.sbatch          # 成对 B 再 A
  sbatch -w <节点> -J lyl-vpd-aNN -t 03:00:00 --export=ALL,ONCE=p4d4-pair3,SCORE_ENV=aNN,PAIR_SIDES=B run_once.sbatch   # 已有同节点基线时只跑 B
  ```
  一轮计分约 2 h（部署 15–25 min，新节点首次要 JIT），门禁约 15 min。节点之间 a8 配置差 < 0.5%，但个别节点 prefill 明显慢（v3-113），所以比较都用同节点成对。
- 监控：`cp $B/scripts/watch_once.sh /tmp/watch_once.aNN.sh && bash /tmp/watch_once.aNN.sh <作业号> p4d4-pair3 p4d4-<S_TAG> $B/results/aNN/B`
  （后台运行，`notify_on_output` 匹配 `WATCH <作业号> .*(FAIL|CRASH|STALL|GONE|scored|gate_ok)`）；从 `/tmp` 的副本运行，避免改脚本时影响正在跑的监控。
  只门禁的多侧作业用一个简单的循环看 `results/aNN/<侧>/gate.txt`（本会话 a21 的写法见 README）。
- 每轮结束后的分析在 `results/aNN/<侧>/`（score、window_tput、dp_balance、ttft_breakdown、steplog_summary、dp_imbalance、trace-cat-dp1、kernel-dist-dp1、trace-gaps）。
  中途比较：`scripts/decode_gen_rate.py <decode 日志> HH:MM:SS HH:MM:SS`（同测量分钟的 decode 生成速度）；
  prefill 排队构成：prefill 日志 `Engine 000:` 行的 Running / Waiting / Deferred / KV fetch（README"TTFT 拆解"）。
- 基准 / 单元测试（1 GPU 小作业）：`once/bench-topk-prod*.sh`、`scripts/bench_topk_prod.py`（分布 randn / zsparse / zmid / mixed / ar / segrelu）、
  `scripts/test_tritontopk.py`、`scripts/sim_topk_est.py`（CPU 仿真采样阈值的失败率）。

## 6. 环境问题（需要知道）

- **v3-186 被 drain**（11-24 05:44，本地盘满，`prepare stepd directory: No space left on device`）：可能是 a4 的 GEMM 调优把大量 FlyDSL kernel 编进了本地 docker 卷
  `vpd-aiter-jit-*`；需要管理员清理。以后调优用一次性 JIT 卷（`TUNE_JIT_VOL`，`scripts/tune_decode_gemm.sh`）。
- v3-176：另一个会话的残留进程占着显存（GPU 0–3 只剩 100 GiB），作业会按空闲显存检查直接退出；v3-113：prefill 明显慢（prefill 排队 2.31 s、计算 1.11 s）。都在排除列表里。
- 偶发：节点"unfinished job cgroup cleanup"导致启动失败、重排 5 次后被挂起（`JobHoldMaxRequeue`）——取消后钉到空闲节点重提；a12 被外部取消过一次（原因不明）。
- prefill 偶发卡死（a4：CPU 层读回的 `hipMemcpyBatchAsync`），出现时该轮作废。

## 7. 复盘要点（详见 README"复盘"）

- 通道周期约 29.5 s = 客户端思考 28% + TTFT 13% + decode 59%；step 时间对吞吐的弹性只有 0.3–0.6，TTFT 每秒约 3%。
- 上 P4D4 前用能复现线上行为的输入验证（Triton top-k 第一版在不相关的合成数据上通过、线上连续失败两轮）；同时看下游 kernel（乱序输出曾拖慢稀疏 MLA）。
- 低于 1% 的项不单独占 P4D4；两个名额并行；成对作业不钉节点、排除坏节点。
