# 交接：GLM-5.2 P4D4（Infera + vLLM）继续优化（2026-11-24 16:55 写，19:00 更新）

接手的 agent 请先读本文，再读同目录 `README.zh-CN.md` 最上面的"总览"、"复盘"、"TTFT 拆解"、a18 / a19 两节。计划在 `../vllm-pd-opt-plan-20261123/PLAN.zh-CN.md`。

## 0000. 11-26 03:20 用户方向（覆盖下面计划里的重复 / 确认运行）

- **不做确认性的重复运行**：已有分数的配置不再重跑（a40 确认对已取消）。新的最好配置 = **a39 D 58,901**（a25 + 目标层 dense FP8 `S_DFP8=1` + MXFP4 indexer `S_IDXFP4=1` + 引擎关 THP `S_THPOFF=1`，`once/score-a39-D.env`），后续每项只跑 **B 一侧**，对照 a39 D。
- **执行表剩余项优先**：第 5 项 CPU 层覆盖层 a41（312975，只跑 B）进行中；第 7 项（#66799 MoE 填充行跳过）、第 8 项（三次 all-gather 合一）由子 agent 写代码（只写不提交，作业由主会话提交，新脚本 `once/p4d4-pair10.sh`）。
- **KFD 撤队列不根治**，只保证不影响评估：每次运行用 pair9 的主机遥测检查计分窗口内的撤队列（`scripts/kfd_evict_summary.py`），有则修正 / 作废该轮，不重跑基线。

## 000. 11-25 17:00 交接（Claude Code 接手后的状态；新会话先读这一节，再读 00 和 `ITEMS-STATUS.zh-CN.md` 第 8 节）

**规则更新**：① "合计 ≤ 4 个已提交"不是集群限制，作业会排队，需要时直接提交（超限显示 `QOSMaxJobsPerUserLimit` / `QOSGrpNodeLimit`）；② 改数值的项（#61985 MXFP4 indexer、dense 在线 FP8）不用等批准，按门禁 / eval 正确性 + 成对性能决定采用与否；
③ 提交用 `git commit -s`，结尾加 `Co-Authored-By: Claude ...`（不再是 Cursor）；④ 不要改正在执行的脚本（`$R/once/p4d4-pair5/6/7.sh` 各有作业在用时，新版另存：pair5 → pair6 → pair7）；⑤ 子 agent 只写代码，作业由主会话提交（曾发生过子 agent 擅自提交作业）。

**当前最好 / 候选（都是单次，tok/s/GPU）**
- 推荐基线 a25 配置：v3-037 56,342 / 56,448；v3-137 56,321（a32 A）、55,222（a33 A，有 2 次 prefill 停顿）；v3-181 55,788。**同节点同配置有无停顿差 1.9%。**
- **a34 B（第 9b 项 dense 在线 FP8，`S_DFP8=1`，pair7）= 58,065（v3-026）**，各项延迟都更好、步长 −6%、门禁干净、真实接受 −0.5%；同节点 A 在 a34（311635）里，约 17:18 出分后下结论。
- a33（第 9 项 MXFP4 indexer，`S_IDXFP4=1`，pair6）B = 54,961（一条长会话 `16b968359e79` 只跑了 8 个请求，去掉后 +0.3–0.8%，接受 −0.9%）；a35（311735）重复 B 验证，约 17:50 出分。
- 第 1、2 项（Mooncake 参数）不采用；第 3 项（调度覆盖层）没有收益、不采用；见 README 11-25 的各条更新。

**比较分数前必查**：① 长会话 `16b968359e79` 是否跑完（`profile_export.jsonl` 里该 `source_trace_id` 的请求数应为 95–96，决定约 3.4% 分数）；② 测量期有无 prefill 停顿（`scripts/engine_stalls.py`；一半的 v3-137 运行有，约 −1–2%）。

**监控**：会话断开监控就停。重挂用 `/tmp/w34.sh`（= `scripts/watch_once.sh` 的副本）：`bash /tmp/w34.sh <作业号> <ONCE 名> <运行目录名> $B/results/aNN/<侧>`，再加上测量期 prefill 空档检测（README / 本会话写法：读 prefill 日志里 `Engine 000:` 行的时间间隔，只在测量开始后报 > 25 s 的空档）。

**接下来 10 小时的计划（按顺序，自主决策）**
1. a34 A 出分 → 第 9b 项结论（采用与否）；若采用，做更大的正确性评估（全量 GSM8K 等，与 BF16 配对）。
2. a35 B 出分 → 第 9 项结论（那条长会话是否再次停止；停止则怀疑 50–80 万 token 下的 MXFP4 打分，要查）。
3. 若 9b / 9 都采用：做组合（`S_DFP8=1 S_IDXFP4=1`，a36，pair7，成对，先门禁），再对最好的配置重复一次给出波动。
4. 调查 prefill 停顿（约 30–70 s，v3-137 上一半运行；怀疑 CPU 层读回 `hipMemcpyBatchAsync` 卡住）：用第 5 项覆盖层（`S_KVOFF=1` 或 `instr`）+ 停顿检测，看是否能消除。
5. 执行表剩余项：4（#67656 + Mooncake 重编，MC#5680）、6（准入 / SPF）、7（#66799）、8（三次 all-gather 合一）、10、11；按"估计收益 ≥ 0.5% 才占作业"。
6. 每一步结果写进 `README.zh-CN.md`（新的在上）+ 更新 `ITEMS-STATUS.zh-CN.md` 对应行 + `git commit -s` 并推送。

## 00. 11-25 06:30 交接（新会话先看这里，再看 `SUMMARY-20261125.zh-CN.md` 和 `ITEMS-STATUS.zh-CN.md`）

- **推荐配置 = a25 配置**（`once/score-a25-B.env`：a8 配置 + Triton top-k 第三版 + prefill `--max-num-batched-tokens 4096 --long-prefill-token-threshold 2048`）。
  v3-037 两次 56,342 / 56,448（比同节点 a8 配置 +5.2%），v3-239 成对 +4.9%，比 sglang +5.7%。详细见 `SUMMARY-20261125.zh-CN.md`。
- **工作方式**：按 `ITEMS-STATUS.zh-CN.md` 第 8 节的执行表逐项做，做完一项更新该行的"状态 / 结果"，实验细节照旧写进 `README.zh-CN.md`（新的在上），
  数据进 `results/aNN/`，及时提交。第 9 节是 InferenceX 规则（改数值的项要 eval + 进上游；草稿不能降精度 / 减 FLOPs；不允许引擎补丁），
  第 10 节是 KV 传输的结论（不是主瓶颈；HIP IPC 在发送端有计算时比 RDMA 慢得多，不做）。
- **运行中**：a31（310832，v3-181，`once/p4d4-pair3.sh`，成对：B = a25 配置 + `MC_RDMA_SIGNAL_INTERVAL=32 MC_MAX_WR=4096`（prefill、decode 都设，`once/score-a31-B.env`），
  A = a25 配置（`once/score-a31-A.env`））。06:19 开始部署，prefill 已起来；B 约 07:40、A 约 10:00 出分。原基线节点 v3-037 / v3-239 被他人一天期作业占用。
  监控进程随会话结束会停，新会话要重挂：
  `cp $B/scripts/watch_once.sh /tmp/w.a31B.sh && bash /tmp/w.a31B.sh 310832 p4d4-pair3 p4d4-a31B-mcsig32wr4096-pmnbt4096-plongthr2048-topkv3-a8cfg-3600 $B/results/a31/B`
  （A 侧把 run 名换成 `p4d4-a31A-pmnbt4096-plongthr2048-topkv3-a8cfg-3600`、结果目录 `results/a31/A`），`notify_on_output` 匹配 `WATCH 310832 .*(FAIL|CRASH|STALL|GONE|scored)`。
- **看 a31 的效果**：除了分数和 TTFT，重点比 B / A 的 KV 拉取：prefill 日志 `MooncakeConnector=` 行的 `Avg xfer time (ms)`（a25 配置约 203 ms），
  decode 侧 `scripts/engine_queue.py <decode 日志> <测量开始+300s> <测量结束>` 的 `fetch_prog`（约 1.76，除以 req/s 3.66 ≈ 0.48 s）。
- **新工具**：`scripts/engine_queue.py`（引擎排队构成）、`scripts/engine_stalls.py`（prefill 停顿）、`scripts/gate_accept_fit.py`（门禁接受去掉失控回答）、
  `scripts/ana_on_node.sh`（需要 numpy 的分析在自己作业的节点上跑，例如 `ttft_analyze.py`）。`once/p4d4-pair4.sh` 支持每侧的 GEM / Mooncake 协议（目前不用）。
- **注意**：坏节点再加 v3-071（decode 慢 30%）、v3-157（启动被杀），提交时一起排除；`logs/engines/*.log` 会被别的部署覆盖，失败原因看运行目录里的 `vpd-*.stream.log.gz`；
  名额：~~自己 ≤ 2 个运行中、合计 ≤ 4 个已提交~~ **11-25 12:00 用户更正**："合计 ≤ 4 个"不是集群限制，作业提交后能排队（超出时显示 `AssocMaxJobsLimit` / `QOSGrpNodeLimit` 等待），需要时直接提交。

## 0. 19:00 更新（下面第 2–4 节是 11-24 16:55 的原文，过时处已标注）

- **11-25 04:40：阶段性总结见 `SUMMARY-20261125.zh-CN.md`**。推荐 a25 配置（`once/score-a25-B.env`），v3-037 上两次平均 56,395（比 a8 配置 +5.2%），
  v3-239 上成对 +4.9%，比 sglang +5.7%。现在没有运行中的作业；下一阶段是改代码（KV 传输路径、prefill 调度等，见总结第 7 节）。

- **20:45：目前最好 a22 = 56,004**（v3-037，单次）= a8 配置 + prefill 阈值 4096 + Triton top-k 第三版，同节点比 a8 配置 +4.5%、比 sglang +5.0%（README a22 节）。
  a24 B（同配置，v3-111）53,462：这台 prefill 慢，等 A 侧成对。a25（v3-037，先影子比对门禁、再计分"a22 + prefill 分块 4096 + 阈值 2048"）进行中，约 23:00 出分。
- **21:05：影子比对（a25 S 侧）——Triton top-k 在 716 万行生产 logits 上 0 个不精确的行**，与 aiter / vLLM 自带只在并列元素上取舍不同（三者彼此差不多）。正确性问题关闭。
- **22:00**：a24（v3-111）成对只 +0.1%（53,462 对 53,418），但 B 侧第 49 min 有一次偶发的 91 s prefill 停顿（之前 +3.2%），不作结论；
  第三台节点成对 a26（309894，v3-071，B 约 23:30、A 约 01:45）。负载里固定有 664k / 539k / 950k token 的超长未命中 prefill，不开阈值时让 prefill
  停 20–90 s（小请求等 65 s），阈值 4096 基本消掉（README"负载里的超大 prefill"节，`scripts/engine_stalls.py`）。
  下一步：a25 B 出分（约 22:38）后在 v3-037 上跑 a27（decode `--gpu-memory-utilization 0.97`，`once/score-a27-B.env`，prefill 部分取 a22 / a25 中好的）。
- **22:40：目前最好 a25 = 56,342**（v3-037，单次）= a22 + prefill `--max-num-batched-tokens 4096 --long-prefill-token-threshold 2048`
  （同节点比 a8 配置 +5.1%、比 sglang +5.6%，TTFT 均值 2.95 s；比 a22 +0.6% 在波动边缘）。a27（310042，v3-037）= a25 + decode 显存 0.97 进行中，约 01:00 出分；
  a26（309894，v3-071）成对 B 约 23:30、A 约 01:45。另一个会话的 DSV4.1 加了 2 个排队作业，本用户合计 5 个已提交，等降下来再投新的。
- **00:20**：a27（a25 + decode 显存 0.97）= 56,463（+0.2%，波动内；TTFT 均值 2.80 s）。a26 B（v3-071）47,440：**v3-071 decode 慢 30%（读 KV 的 kernel 慢一倍），坏节点，
  以后排除**。a28（310195，v3-037，成对：B = a27 配置重复、A = a25 配置重复）进行中，B 约 02:15、A 约 04:30。监控 `/tmp/watch_once.a28B.sh`、`/tmp/watch_once.a26A.sh`。
  推荐配置候选：a25（`once/score-a25-B.env`），若 a28 显示显存 0.97 稳定不差则用 a27（`once/score-a27-B.env`）。

- **成对结果**（各单次）：a18 Triton top-k 第三版 vs vLLM 自带 +1.4%（v3-231）；a19 prefill 阈值 4096 vs a8 配置 +1.6%（v3-037）；
  a20 prefill 分块 4096 = 54,036，同节点 vs a8 配置 +0.8%（降的是 prefill 计算 / 流水线 0.82 → 0.58 s，阈值降的是排队 1.15 → 0.70 s，两者互补）。
- **Triton top-k 的"真实接受低 1.2%"是门禁假象**（README a21 节）：门禁 temperature 0 也不可复现，每次 1–4 个回答随机失控到 16k 上限
  （草稿接受 3.97、各占约 10% 的 token）；去掉后每 token 接受 不开 3.352 / Triton 3.356。以后比门禁接受用 `scripts/gate_accept_fit.py`。
- **TTFT 拆解更正**：prefill 的 Deferred 是 0.74（不是 1.89，那是 decode 的）→ 等准入约 0.79 s 是大头，CPU 层读回约 0.21 s（`scripts/engine_queue.py`）。
- **运行中**：a22（309600，钉 v3-037，只 B：a8 + 阈值 4096 + Triton top-k，先门禁，约 21:15 出分，与 a19 A/B、a20 B 同节点比）；
  a24（309580，钉 v3-111，成对：B = a22 同配置不门禁约 21:00，A = a8 配置约 23:15）。监控：`/tmp/watch_once.a22.sh`、`/tmp/watch_once.a24B.sh`。
- **待办**：(1) a22 / a24 出分后写组合的结论（README 总览 + 新节）；(2) a25 = prefill 分块 4096 + 阈值 2048（若 a22 证明 Triton 能叠加，就在 a22 配置上改 prefill，
  钉 v3-037 只跑 B 与 a22 比；配置草稿 `once/score-a25-B.env`）；(3) a23 影子比对（`once/score-a23-B.env`，读线程已修，单元测试通过）有空名额再跑，
  看 decode 日志 `[vpd-topk-shadow]` 行里 t_val_bad 是否为 0；(4) 对最好的配置再重复一次。
- 名额：本用户实际同时跑了 3 个（含 DSV4.1 的），仍按"自己 ≤ 2 个运行中、合计 ≤ 4 个已提交"来。

## 1. 目标与用户的要求

- 目标：在 `dev/pd_opt/vllm_glm52_agentx` 分支上继续推高 P4D4（MI355X，8 卡）AgentX C104 3600 s 的 **tok/s/GPU**，同时降低 ITL / 提高 interactivity。
  对齐 sglang P4D4（53,351）只是初步目标（用户 11-24 05:50）。
- 用户 14:05 选定的主攻方向：**TTFT 与 decode 准入**（不改数值的调度类改动优先）；~~dense 层在线 FP8 未获批准，不要做~~ **11-25 07:55 用户更正**：改数值的项（dense 层在线 FP8、#61985 MXFP4 indexer K cache）不用等批准，按 InferenceX 规则可做的就做，根据门禁 / eval 的正确性和成对性能结果决定是否采用。
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
- ~~**Triton top-k 有未解决问题**：门禁真实接受长度四次都低约 1.2%~~ **18:05 查清**：是门禁的统计假象（失控到 16k 上限的回答个数随机，
  不开时每次 2 个、Triton 每次 1 个；去掉后每 token 接受 3.352 对 3.356，README a21 节，`scripts/gate_accept_fit.py`）。a23 影子比对补直接证据。
- a18 的 Triton 第三版：`scripts/vpd_triton_topk_decode.py`（2048 段 × 4 采样、OVER 4、CAND 16,384、同分值保护、基数直方图慢路径、每行升序排序 `_sort_rows`），
  覆盖层 `scripts/make_tritontopk_overlay.py`（`VPD_TOPK_IMPL=triton|native`、`VPD_TOPK_SCOPE=all|target|draft`、`VPD_TOPK_SORT=0` 关排序）。

## 3. 进行中 / 排队的作业（16:55）

| 作业 | 内容 | 状态 / 下一步 |
|---|---|---|
| 309187 a18 | `p4d4-pair2.sh` SCORE_ENV=a18：B = Triton 第三版（已完成 54,484），A = vLLM 自带 top-k，v3-231 | **完成（17:02）**：A = 53,730，成对 +1.4%（README a18 节） |
| 309209 a19 | `p4d4-pair3.sh` SCORE_ENV=a19：B = long-prefill-threshold 4096（已完成 54,449），A = a8 配置，v3-037 | **完成（17:19）**：A = 53,610，成对 +1.6%（README a19 节） |
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
- **11-24 晚新增的坏节点（提交时加进 `-x`）**：v3-071 decode 慢约 30%（读 KV 的 kernel 慢一倍，功率 / 频率正常，a26）；v3-157 作业 12 s 内被 SIGKILL（未开始，a26 第一次提交）。
  v3-111 正常（a24 A / B 的 decode 与 v3-037 一致），只是 a24 B 碰到过一次偶发 prefill 停顿。
- 新节点第一次跑作业时要先构建镜像（Mooncake 编译）约 10–15 min，`once-<作业>.out` / `build-<节点>-*.log` 可看进度。
- prefill 偶发卡死（a4：CPU 层读回的 `hipMemcpyBatchAsync`），出现时该轮作废。

## 7. 复盘要点（详见 README"复盘"）

- 通道周期约 29.5 s = 客户端思考 28% + TTFT 13% + decode 59%；step 时间对吞吐的弹性只有 0.3–0.6，TTFT 每秒约 3%。
- 上 P4D4 前用能复现线上行为的输入验证（Triton top-k 第一版在不相关的合成数据上通过、线上连续失败两轮）；同时看下游 kernel（乱序输出曾拖慢稀疏 MLA）。
- 低于 1% 的项不单独占 P4D4；两个名额并行；成对作业不钉节点、排除坏节点。
