# 交接（2026-10-15 03:45 UTC）—— 新会话先读这个

## 目标与当前结论
- 目标：升级镜像（sglang v0.5.21 + Mooncake 跨 rank 修复）后，3P(TP4)1D(TP8) GLM-5.2-MXFP4 在 AgentX 上的最高吞吐（tok/s/GPU）；
  用户口径的 interactivity 约束是 P50 ∈ [50,100]，但 10-14 起用户要求"以推高吞吐为目标"。
- **目前最高：3P1D C176 = 30,562 tok/s/GPU**（t4d-tok4；P50/P90 interactivity 48.9/44.7）。详见 `RESULT.zh-CN.md` §0/§3（表格由
  `scripts/tune_report.py` 生成）。最优配置 = router R1 + P 会话亲和 + R2；prefill mem 0.88 + HiCache 400/200 GB/rank；decode
  `--tokenizer-worker-num 4`（`config/best.sh`，复现：`REPRODUCE.zh-CN.md`）。
- 2P(TP4)1D C176 = 28,900 tok/s/GPU（16 卡，−5.4%，prefill 成瓶颈 + 实例间失衡）。C192（3P1D）因挂起无有效结果（部分数据未超过 C176）。

## 栈当前状态（10-15 05:21，ROUND=r2，容器前缀 glm52-llying-3p1d-r2-）
- decode-0（138，mem **0.92**，tok4，日志名 `decode-0.after-mem092.log`）、prefill-0（135，400 GB HiCache）、prefill-1（137 **card4–7**）、
  prefill-2（137 card0–3，05:14 重启，**`HSA_NO_SCRATCH_RECLAIM=1`**）在运行；router 在 137。`config/round.r2.sh` = `config/stage-mem092-nsr.sh`。
- **10-15 11:01 状态**：队列在跑 t8（3P1D + prefill_complete + write_back，`config/stage-pcwb.sh`；第 37–42 步，见 PLAN-TUNING 末节"10:50 执行顺序"）；
  sampler、hang 监视器已按加固后的版本重启；编排脚本已做防 `/home` 写满加固（STATUS 11:00 条）。
- 10-15 09:55 状态（已过时）：第 34–36 步（warm-mem092b、t6-mem092 C176 = 30,291、C192 = **31,829，目前最高**）已完成；07:57 共享 `/home` 写满，
  队列驱动、轮次脚本、hang 监视器全部退出（sampler 自 14 日 22:00 起就已退出），**现在没有任何编排进程在跑**；C192 已手工补救（`results/t6-mem092/c192/RESCUED.txt`），
  `queue.done` = 36。重启编排前先做 EDQUOT 加固（见 STATUS 末条）。
- 挂起监视器 `scripts/hang_watch.py`（`scripts/start_hang_watch.sh` 幂等启动，日志 `runtime/runs/r2/tune/hang_watch.log`）：
  任一 prefill 有请求但 10 分钟无进展即自动 `scripts/hang_capture.sh`，结果在 `runtime/runs/r2/tune/hang-<实例>-<MMDD-HHMM>/`。
- 日志 follower 05:00 已用 `scripts/log_follow.sh ROUND=r2` 恢复（之前 21:59 起全部退出）。

- 原始运行数据：服务端日志等在 `runtime/`（gitignore）；各轮客户端完整输出（含服务端指标导出）在
  `/shared_nfs/liyingli/bench_agentx/3p1d-tp4-tp8-v0521-20261012/node137-runs-r2/`（10-15 16:15 从 137 本地盘复制）。

## 未完成 / 下一步（按优先级）
1. **prefill-2 挂起**：排查结论和 A/B 方案见 `RESULT.zh-CN.md` §4.12。主要假设是 prefill 的 `HSA_NO_SCRATCH_RECLAIM=0` 在显存紧张时让 ROCr scratch
   分配进入 retry 等待，与集合通信互锁；IPC socket 撞名已排除。现在只有 prefill-2 改成 1，prefill-0/1 作对照。
   - 若 prefill-0/1 挂起：看自动抓取的 `rocgdb-queues-*.txt`（哪个队列有待执行包却没有活跃 dispatch、哪些通信 kernel 在跑），可作为直接证据；
   - 若 prefill-2 挂起：假设不成立，同样看抓取结果；
   - 若跑满几轮都不挂：把三个 prefill 都改成 1（注意早期 `OUT_OF_RESOURCES` 风险，见 `scripts/bench-harness/issue.md` §3.4）。
2. decode mem 0.92 的 3P1D C176 / C192（`config/stage-mem092.sh` 已就位，需先拉起 prefill-2：`restart_prefills.sh INSTANCES=prefill-2`，
   但它可能再挂）→ 看 decode KV 扩大后峰值能否超过 30,562。
3. prefill_complete（需 prefill HiCache 改 write_back；可释放 decode 中"等 KV 传输"请求占的 7–12% KV；先看 mem 0.92 有没有收益再决定，
   理论分析见 `RESULT.zh-CN.md` §4.7）；`index_share_for_mtp_iteration=true`（抬 decode 每 rank 上限，需真实接受率门禁验证正确性）；
   1P(TP8)1D（每 rank GPU KV 约 +29%，但需 137 整机）。
4. **按 InferenceX 官方协议复测最优配置**（`RESULT.zh-CN.md` §5）：官方要求每个并发点重新部署、不 flush、不复用在线服务；我们的 30,562 是在复用的栈上、
   tok4 重启后第三轮测得的，而重启后首轮 decode 每步慢 3–17%。做法：整栈重启 → 直接测 C176（只用 AgentX 自带预热），客户端最好换成官方的
   `agentx-harness==1.0.9`。另外，`stream-interval` 是各 recipe 自选的参数（官方 GLM-5.2 sglang 多机 recipe 用 30 或 60，我们是 1），值得作为单独一轮测试。

## 关键数据口径（易错）
- interactivity = 1 / aiperf Full-Response ITL（每请求平均 token 间隔）的 P50/P90。
- decode 容量：每个在跑请求实际占 KV ≈ 18–21 万 token（按停留步数加权，长请求输出也长），不是平均输入 12.5 万；
  "等 KV 传输"的请求占 decode KV 7–12%（随 prefill 延迟增长），"等预分配"的请求不占 KV。
- 每 rank decode 输出在在跑 8–14 个时平在 650–720 tok/s；ITL≈20 ms 对应每 rank 约 10–11 个请求（与 P8D4 一致）。
- TTFT 分段：`scripts/sample_window.py samples.jsonl START END`；prefill 服务不报 TTFT，decode 的 `time_to_first_token` 约等于客户端 TTFT。

## 操作要点与踩过的坑
- 一轮 = `scripts/tune_round.sh`（flush → router 带开关 → AgentX → `tune_post.sh` 发布）；队列 `scripts/start_queue.sh ROUND=r2`（幂等）。
- 引擎重启顺序：**先 decode 后 prefill**（只重启 decode 会让部分传输通道失效）；重启后必须跑 `scripts/transfer_smoke.py`（后台跑再轮询）。
  重启后第一轮 decode 偏慢 3–17%，先跑 `warmup_round.sh`。HiCache 栈停后显存回收 10–30 分钟（135 的 400 GB 栈最慢）。
- 巡检：`scripts/tune_watch.py r2 10`、`scripts/hang_check.py r2 180`；census 要区分窗口内与窗口结束（取消）条目。
- **不要在命令里用 `pgrep -f` 匹配自己命令行里出现的字符串**（会自匹配）；工具层面偶尔会把同一条命令执行两次 → 启动类操作必须幂等、
  单独先检查再启动；**长时间阻塞的命令会被中断并导致会话停摆** → 长任务一律在节点后台执行、本地只用短命令轮询。
- 运行中的脚本不要原地编辑（bash 增量读取；NFS 上 mv 替换会 stale handle）。
- `/home` 是共享 10 TB 卷，曾写满导致 follower/编排退出；拷回已排除 `server_metrics_export.json`，每轮要求 ≥ 20 GB。
- 只动名字含 `llying` 且前缀匹配的容器；135 GPU1、138/136 上有其他租户。
- 过程记录：`operations/STATUS.zh-CN.md`（时间线最全）；计划与协议：`PLAN-TUNING.zh-CN.md`。
